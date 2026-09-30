"""
pmu_server.py
--------------
Ties everything together: generates the signal, computes phasors,
builds frames, and sends them over a TCP socket to any connected
clients at a precise reporting rate (default 50 frames/second).

Every generated frame is also pushed into a CircularBuffer first. Each
connected client gets its own sender thread that replays any backlog it
missed (e.g. if it just reconnected) and then keeps streaming live
frames. This means a client can disconnect, the server keeps running
and buffering, and the client catches up automatically when it comes
back -- it doesn't have to be the main loop's problem.

Outputs: a live data stream of binary frames over the network.
"""

import socket
import threading
import time
import argparse

from signal_generator import SignalGenerator
from phasor_calculator import compute_three_phase_phasors
from frame_builder import build_frame
from circular_buffer import CircularBuffer


class PMUServer:
    def __init__(self, host="127.0.0.1", port=8765, idcode=1,
                 nominal_freq=50.0, sample_rate=1200, report_rate=50,
                 buffer_seconds=5, noise_level=0.01):
        self.host = host
        self.port = port
        self.idcode = idcode
        self.report_rate = report_rate

        self.generator = SignalGenerator(
            nominal_freq=nominal_freq, sample_rate=sample_rate,
            noise_level=noise_level
        )
        self.buffer = CircularBuffer(maxsize=int(buffer_seconds * report_rate))

        self._clients = []          # list of (conn, addr, last_seq_sent)
        self._clients_lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        listener = threading.Thread(target=self._accept_loop, daemon=True)
        listener.start()

        print(f"[pmu_server] PMU id={self.idcode} listening on "
              f"{self.host}:{self.port}, reporting at {self.report_rate} fps")
        self._generation_loop()

    def stop(self):
        self._running = False

    # ---- networking ----

    def _accept_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(5)
        srv.settimeout(0.5)

        while self._running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f"[pmu_server] client connected: {addr}")
            client_thread = threading.Thread(
                target=self._client_sender, args=(conn, addr), daemon=True
            )
            client_thread.start()

        srv.close()

    def _client_sender(self, conn, addr):
        """
        One thread per client: first replay any backlog already sitting
        in the buffer (so a reconnecting client catches up instantly),
        then keep tailing new frames as the generation loop produces
        them.
        """
        last_seq_sent = 0
        try:
            # Catch-up replay of whatever is currently buffered
            backlog = self.buffer.get_all()
            for seq, frame in backlog:
                conn.sendall(frame)
                last_seq_sent = seq

            # Live tail
            while self._running:
                new_frames = self.buffer.get_since(last_seq_sent)
                if not new_frames:
                    time.sleep(1.0 / self.report_rate / 2)
                    continue
                for seq, frame in new_frames:
                    conn.sendall(frame)
                    last_seq_sent = seq
        except (BrokenPipeError, ConnectionResetError, OSError):
            print(f"[pmu_server] client disconnected: {addr} "
                  f"(network down -- frames will keep buffering for replay)")
        finally:
            conn.close()

    # ---- frame generation ----


    def _generation_loop(self):
        period = 1.0 / self.report_rate
        start = time.perf_counter()
        frame_count = 0

        try:
            while self._running:
                target_time = start + frame_count * period
                now = time.perf_counter()
                if target_time > now:
                    time.sleep(target_time - now)

                t, cycle = self.generator.generate_cycle()
                phasors = compute_three_phase_phasors(
                    cycle["Va"], cycle["Vb"], cycle["Vc"]
                )
                frame = build_frame(phasors, idcode=self.idcode, timestamp=time.time())
                self.buffer.push(frame)

                frame_count += 1
                if frame_count % (self.report_rate * 5) == 0:
                    stats = self.buffer.stats()
                    print(f"[pmu_server] {frame_count} frames sent | "
                          f"buffer={stats['current_size']}/{stats['max_size']} | "
                          f"dropped={stats['total_dropped']}")
        except KeyboardInterrupt:
            print("\n[pmu_server] shutting down")
            self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMU simulator server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--idcode", type=int, default=1)
    parser.add_argument("--report-rate", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args()

    server = PMUServer(host=args.host, port=args.port, idcode=args.idcode,
                        report_rate=args.report_rate, noise_level=args.noise)
    server.start()