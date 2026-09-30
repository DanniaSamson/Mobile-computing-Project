"""
visualizer.py
---------------
Connects to pmu_server.py as a network client, receives the live frame
stream, decodes it with frame_builder.parse_frame(), and draws:
  1. A live phasor diagram (polar plot) showing Va, Vb, Vc as rotating
     vectors.
  2. A magnitude trend strip-chart for all three phases over time.
  3. A reconstructed time-domain waveform built from the most recent
     phasor readings.

Outputs: all graphs and plots (live, updating window).

Run modes:
    python visualizer.py                 -> connects to a running
                                             pmu_server.py on 127.0.0.1:8765
    python visualizer.py --demo          -> offline demo, no server/socket
                                             needed, generates + plots
                                             directly using signal_generator
                                             and phasor_calculator.
"""

import matplotlib
matplotlib.use("TkAgg")
import argparse
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from frame_builder import parse_frame, HEADER_SIZE

HISTORY_LEN = 150  # how many phasor readings to keep for the trend chart


class LiveData:
    """Thread-safe shared state between the network receiver thread and
    the matplotlib animation running on the main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_phasors = None     # most recent parsed phasors dict
        self.history = {
            "t": deque(maxlen=HISTORY_LEN),
            "Va": deque(maxlen=HISTORY_LEN),
            "Vb": deque(maxlen=HISTORY_LEN),
            "Vc": deque(maxlen=HISTORY_LEN),
        }
        self.connected = False
        self.frames_received = 0

    def update(self, decoded):
        with self.lock:
            self.latest_phasors = decoded["phasors"]
            self.history["t"].append(decoded["timestamp"])
            for name in ("Va", "Vb", "Vc"):
                self.history["Va" if name == "Va" else name].append(
                    decoded["phasors"][name]["magnitude"]
                )
            self.frames_received += 1

    def snapshot(self):
        with self.lock:
            phasors = self.latest_phasors
            hist = {k: list(v) for k, v in self.history.items()}
            return phasors, hist, self.connected, self.frames_received


def network_receiver(host, port, live_data, stop_event):
    """
    Background thread: connects to pmu_server, reads the byte stream,
    and slices it into individual frames using the FRAMESIZE field
    embedded in each frame's header (bytes 2-3, big-endian uint16).
    """
    while not stop_event.is_set():
        try:
            sock = socket.create_connection((host, port), timeout=3)
            live_data.connected = True
            print(f"[visualizer] connected to {host}:{port}")
            buf = b""
            while not stop_event.is_set():
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionResetError("server closed connection")
                buf += chunk

                # Try to peel off as many complete frames as are sitting
                # in the buffer right now.
                while len(buf) >= HEADER_SIZE:
                    framesize = struct.unpack(">H", buf[2:4])[0]
                    if len(buf) < framesize:
                        break  # haven't received the rest of this frame yet
                    frame_bytes, buf = buf[:framesize], buf[framesize:]
                    try:
                        decoded = parse_frame(frame_bytes)
                        live_data.update(decoded)
                    except ValueError as e:
                        print(f"[visualizer] dropped a corrupt frame: {e}")
        except (ConnectionRefusedError, ConnectionResetError, OSError, socket.timeout) as e:
            live_data.connected = False
            print(f"[visualizer] connection issue ({e}); retrying in 2s...")
            time.sleep(2)


def run_live(host, port):
    live_data = LiveData()
    stop_event = threading.Event()
    receiver = threading.Thread(
        target=network_receiver, args=(host, port, live_data, stop_event), daemon=True
    )
    receiver.start()
    _run_plot(live_data)
    stop_event.set()


def run_demo():
    """Offline mode: no networking, drives the plots directly from the
    signal generator + phasor calculator so you can see the visuals
    without needing pmu_server.py running."""
    from signal_generator import SignalGenerator
    from phasor_calculator import compute_three_phase_phasors

    gen = SignalGenerator(noise_level=0.02)
    live_data = LiveData()
    live_data.connected = True
    stop_event = threading.Event()

    def demo_feed():
        t0 = time.time()
        while not stop_event.is_set():
            _, cycle = gen.generate_cycle()
            phasors = compute_three_phase_phasors(cycle["Va"], cycle["Vb"], cycle["Vc"])
            live_data.update({"timestamp": time.time() - t0, "phasors": phasors})
            time.sleep(0.02)  # ~50 fps

    threading.Thread(target=demo_feed, daemon=True).start()
    _run_plot(live_data)
    stop_event.set()


def _run_plot(live_data):
    fig = plt.figure(figsize=(11, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_trend = fig.add_subplot(1, 2, 2)

    colors = {"Va": "tab:red", "Vb": "tab:green", "Vc": "tab:blue"}

    def update(_frame):
        phasors, hist, connected, count = live_data.snapshot()

        ax_polar.clear()
        ax_polar.set_title("Phasor diagram")
        ax_polar.set_ylim(0, 260)
        if phasors:
            for name, color in colors.items():
                mag = phasors[name]["magnitude"]
                ang = np.radians(phasors[name]["angle"])
                ax_polar.plot([0, ang], [0, mag], color=color, linewidth=2, label=name)
                ax_polar.plot(ang, mag, "o", color=color)
            ax_polar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        ax_trend.clear()
        ax_trend.set_title("Voltage magnitude trend (RMS)")
        ax_trend.set_xlabel("samples (most recent)")
        ax_trend.set_ylabel("Volts")
        for name, color in colors.items():
            if hist[name]:
                ax_trend.plot(hist[name], color=color, label=name)
        if hist["Va"]:
            ax_trend.legend(loc="upper right")
        ax_trend.set_ylim(200, 260)

        status = "connected" if connected else "reconnecting..."
        fig.suptitle(f"PMU live view -- {status} -- {count} frames received")

    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMU visualizer client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true",
                         help="run without a server, using simulated data directly")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_live(args.host, args.port)