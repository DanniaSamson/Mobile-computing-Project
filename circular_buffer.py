"""
circular_buffer.py
--------------------
A fixed-size ring buffer that sits between frame generation and frame
transmission. It always keeps the most recent N frames, even if nothing
is reading from it. This is what lets the PMU server "save frames when
the network is down" and "replay when connection restores" -- a client
that reconnects can ask for everything since the last sequence number
it successfully received, and replay it before resuming live data.

Outputs: buffer stats (current size, total pushed, total dropped, etc.)
"""

from collections import deque
import threading


class CircularBuffer:
    def __init__(self, maxsize=250):
        """
        maxsize : number of frames to retain. At 50 frames/sec, a
                  maxsize of 250 holds 5 seconds of backlog.
        """
        self.maxsize = maxsize
        self._buffer = deque(maxlen=maxsize)  # holds (seq, frame_bytes)
        self._lock = threading.Lock()
        self._next_seq = 1
        self._total_pushed = 0
        self._total_dropped = 0

    def push(self, frame_bytes):
        """
        Add a new frame to the buffer. If the buffer is full, the oldest
        frame is silently discarded (that's the "circular" part) and
        counted as dropped.

        Returns the sequence number assigned to this frame.
        """
        with self._lock:
            was_full = len(self._buffer) == self.maxsize
            seq = self._next_seq
            self._buffer.append((seq, frame_bytes))
            self._next_seq += 1
            self._total_pushed += 1
            if was_full:
                self._total_dropped += 1
            return seq

    def get_since(self, last_seq):
        """
        Return all buffered frames with sequence number > last_seq, in
        order, as a list of (seq, frame_bytes) tuples. This is the
        "replay" operation: a reconnecting client passes the last seq
        it successfully received and gets everything it missed that's
        still available in the buffer.

        If the requested backlog is larger than what's still in the
        ring (i.e. some frames were already overwritten while the
        client was disconnected too long), only what's left is returned
        -- those very old frames are simply gone, same as a real PMU
        buffer with finite memory.
        """
        with self._lock:
            return [(seq, frame) for seq, frame in self._buffer if seq > last_seq]

    def get_all(self):
        """Return every frame currently held in the buffer."""
        with self._lock:
            return list(self._buffer)

    @property
    def latest_seq(self):
        with self._lock:
            return self._next_seq - 1

    def stats(self):
        """
        Returns a dict summarizing buffer health -- useful for a
        dashboard or for the visualizer to display network reliability.
        """
        with self._lock:
            oldest_seq = self._buffer[0][0] if self._buffer else None
            newest_seq = self._buffer[-1][0] if self._buffer else None
            return {
                "current_size": len(self._buffer),
                "max_size": self.maxsize,
                "total_pushed": self._total_pushed,
                "total_dropped": self._total_dropped,
                "oldest_seq": oldest_seq,
                "newest_seq": newest_seq,
            }


if __name__ == "__main__":
    # Self-test / demo: simulate the network going down for a while,
    # then "reconnecting" and replaying the backlog.
    buf = CircularBuffer(maxsize=5)

    print("Pushing 3 frames while a client is connected and caught up...")
    for i in range(3):
        buf.push(f"frame-{i+1}".encode())
    client_last_seq = buf.latest_seq
    print(f"Client is caught up through seq {client_last_seq}\n")

    print("Network goes down. Server keeps generating frames (buffer fills, then wraps)...")
    for i in range(3, 9):
        seq = buf.push(f"frame-{i+1}".encode())
        print(f"  pushed seq {seq}")

    print(f"\nBuffer stats while disconnected: {buf.stats()}\n")

    print(f"Network restored. Client replays everything since seq {client_last_seq}:")
    backlog = buf.get_since(client_last_seq)
    for seq, frame in backlog:
        print(f"  replaying seq {seq}: {frame}")

    print(f"\nFinal stats: {buf.stats()}")
    print("(total_dropped > 0 shows some very old frames were overwritten "
          "before the client could catch up -- expected with a finite buffer)")