"""
frame_builder.py
------------------
Packs phasor data into a binary packet structure modeled on the IEEE
C37.118 synchrophasor data frame (the real-world standard used by PMUs
on the power grid). This is a simplified, educational version of that
format -- it follows the same layout ideas (sync word, id, timestamp,
data, CRC) but is not byte-for-byte certified against the published
standard.

Frame layout:
    SYNC      2 bytes   0xAA, 0x01           (start-of-frame marker)
    FRAMESIZE 2 bytes   total length of frame, including CRC
    IDCODE    2 bytes   PMU station ID number
    SOC       4 bytes   timestamp, whole seconds since Unix epoch
    FRACSEC   4 bytes   fractional part of the second, in microseconds
    DATA      24 bytes  3 phasors x (4-byte float magnitude + 4-byte float
                         angle-in-radians) for Va, Vb, Vc
    CHK       2 bytes   CRC-CCITT checksum over everything before it

Outputs: binary packet bytes (and a matching parser to decode them back).
"""

import struct
import time

SYNC_WORD = b"\xAA\x01"
HEADER_FMT = ">2sHHII"   # sync, framesize, idcode, soc, fracsec
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PHASOR_FMT = ">ff"        # magnitude (float32), angle in radians (float32)
PHASOR_SIZE = struct.calcsize(PHASOR_FMT)
CHK_FMT = ">H"
CHK_SIZE = struct.calcsize(CHK_FMT)

PHASE_ORDER = ("Va", "Vb", "Vc")


def crc16_ccitt(data, poly=0x1021, init=0xFFFF):
    """
    Standard CRC-CCITT (XModem variant) checksum, the same style of CRC
    used by the real IEEE C37.118 standard for frame integrity checking.
    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def build_frame(phasors, idcode=1, timestamp=None):
    """
    Build a binary frame from a phasors dict (the output of
    phasor_calculator.compute_three_phase_phasors, angles in degrees).

    idcode    : integer PMU station ID (0-65535)
    timestamp : unix time in seconds (float). Defaults to "now".

    Returns: bytes
    """
    if timestamp is None:
        timestamp = time.time()

    soc = int(timestamp)
    fracsec = int(round((timestamp - soc) * 1_000_000))  # microseconds

    # Build the data section: magnitude + angle(radians) for each phase
    data_bytes = b""
    for name in PHASE_ORDER:
        mag = phasors[name]["magnitude"]
        ang_rad = struct.pack(PHASOR_FMT, mag, _deg_to_rad(phasors[name]["angle"]))
        data_bytes += ang_rad

    framesize = HEADER_SIZE + len(data_bytes) + CHK_SIZE

    header = struct.pack(HEADER_FMT, SYNC_WORD, framesize, idcode, soc, fracsec)
    body = header + data_bytes

    chk = crc16_ccitt(body)
    return body + struct.pack(CHK_FMT, chk)


def parse_frame(frame_bytes):
    """
    Decode a frame produced by build_frame(). Validates the CRC and
    raises ValueError if the frame is corrupt or malformed.

    Returns a dict:
        {
            "idcode": int,
            "timestamp": float (unix seconds),
            "phasors": {
                "Va": {"magnitude": ..., "angle": ...},  # angle in degrees
                "Vb": {...}, "Vc": {...}
            }
        }
    """
    if len(frame_bytes) < HEADER_SIZE + CHK_SIZE:
        raise ValueError("Frame too short to be valid")

    sync, framesize, idcode, soc, fracsec = struct.unpack(
        HEADER_FMT, frame_bytes[:HEADER_SIZE]
    )
    if sync != SYNC_WORD:
        raise ValueError("Bad sync word - not a valid frame")
    if framesize != len(frame_bytes):
        raise ValueError(
            f"Framesize mismatch: header says {framesize}, got {len(frame_bytes)}"
        )

    body = frame_bytes[:-CHK_SIZE]
    (received_chk,) = struct.unpack(CHK_FMT, frame_bytes[-CHK_SIZE:])
    computed_chk = crc16_ccitt(body)
    if received_chk != computed_chk:
        raise ValueError(
            f"CRC mismatch (frame corrupted): got {received_chk:#06x}, "
            f"expected {computed_chk:#06x}"
        )

    data_bytes = frame_bytes[HEADER_SIZE:-CHK_SIZE]
    phasors = {}
    offset = 0
    for name in PHASE_ORDER:
        mag, ang_rad = struct.unpack(PHASOR_FMT, data_bytes[offset:offset + PHASOR_SIZE])
        phasors[name] = {"magnitude": mag, "angle": _rad_to_deg(ang_rad)}
        offset += PHASOR_SIZE

    return {
        "idcode": idcode,
        "timestamp": soc + fracsec / 1_000_000,
        "phasors": phasors,
    }


def _deg_to_rad(deg):
    import math
    return math.radians(deg)


def _rad_to_deg(rad):
    import math
    return math.degrees(rad)


if __name__ == "__main__":
    # Self-test: build a frame, corrupt one for negative testing, parse both
    from signal_generator import SignalGenerator
    from phasor_calculator import compute_three_phase_phasors

    gen = SignalGenerator(seed=1, noise_level=0.0)
    t, cycle = gen.generate_cycle()
    phasors = compute_three_phase_phasors(cycle["Va"], cycle["Vb"], cycle["Vc"])

    ts = time.time()
    frame = build_frame(phasors, idcode=42, timestamp=ts)
    print(f"Frame length: {len(frame)} bytes")
    print(f"Raw bytes: {frame.hex()}")

    decoded = parse_frame(frame)
    print(f"\nDecoded idcode: {decoded['idcode']}")
    print(f"Decoded timestamp: {decoded['timestamp']:.6f} (original: {ts:.6f})")
    for name, p in decoded["phasors"].items():
        print(f"  {name}: magnitude={p['magnitude']:.3f} V, angle={p['angle']:.3f} deg")

    # Corruption test
    bad_frame = bytearray(frame)
    bad_frame[10] ^= 0xFF
    try:
        parse_frame(bytes(bad_frame))
        print("\nERROR: corrupted frame was not detected!")
    except ValueError as e:
        print(f"\nCorruption correctly detected: {e}")