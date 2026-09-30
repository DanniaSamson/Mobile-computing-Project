"""
phasor_calculator.py
---------------------
Takes one cycle of time-domain voltage samples and computes the phasor
representation (magnitude + angle) using the classic single-frequency
Discrete Fourier Transform (DFT) formula used in real PMUs:

    X = (sqrt(2) / N) * sum_{n=0}^{N-1} x[n] * exp(-j * 2*pi*n / N)

    magnitude = |X|   -> this comes out as the RMS value of the signal
    angle     = arg(X) -> phase angle in radians (or degrees)

Outputs: Va, Vb, Vc phasors, each as (magnitude, angle_degrees).
"""

import numpy as np


def dft_phasor(samples, in_degrees=True):
    """
    Compute the phasor (magnitude, angle) of a single phase from one
    cycle of samples using the single-frequency DFT formula.

    samples : 1-D array-like, exactly one electrical cycle of samples
    in_degrees : if True, angle is returned in degrees, else radians

    Returns (magnitude, angle)
    """
    x = np.asarray(samples, dtype=float)
    N = len(x)
    n = np.arange(N)

    # single-frequency DFT at bin 1 (the fundamental), normalized so the
    # magnitude comes out as RMS volts (matches how real PMUs report it)
    X = (np.sqrt(2) / N) * np.sum(x * np.exp(-1j * 2 * np.pi * n / N))

    magnitude = np.abs(X)
    angle = np.angle(X)
    if in_degrees:
        angle = np.degrees(angle)

    return magnitude, angle


def compute_three_phase_phasors(Va_samples, Vb_samples, Vc_samples, in_degrees=True):
    """
    Convenience wrapper: compute phasors for all three phases at once.

    Returns a dict:
        {
            "Va": {"magnitude": ..., "angle": ...},
            "Vb": {"magnitude": ..., "angle": ...},
            "Vc": {"magnitude": ..., "angle": ...},
        }
    """
    result = {}
    for name, samples in (("Va", Va_samples), ("Vb", Vb_samples), ("Vc", Vc_samples)):
        mag, ang = dft_phasor(samples, in_degrees=in_degrees)
        result[name] = {"magnitude": mag, "angle": ang}
    return result


def compute_positive_sequence(phasors_deg):
    """
    Bonus: compute the positive-sequence phasor from the three phase
    phasors using the standard symmetrical-components transform.
    Real PMUs often report this single combined value because it
    represents the "useful" balanced part of the three-phase system.

    phasors_deg: dict like the output of compute_three_phase_phasors()
                 (angles must be in degrees)

    Returns (magnitude, angle_degrees) of the positive sequence component.
    """
    a = np.exp(1j * np.radians(120))  # 120 degree rotation operator

    Va = phasors_deg["Va"]["magnitude"] * np.exp(1j * np.radians(phasors_deg["Va"]["angle"]))
    Vb = phasors_deg["Vb"]["magnitude"] * np.exp(1j * np.radians(phasors_deg["Vb"]["angle"]))
    Vc = phasors_deg["Vc"]["magnitude"] * np.exp(1j * np.radians(phasors_deg["Vc"]["angle"]))

    V1 = (Va + a * Vb + a * a * Vc) / 3.0
    return np.abs(V1), np.degrees(np.angle(V1))


if __name__ == "__main__":
    # Self-test using signal_generator so we can verify the math is right
    from signal_generator import SignalGenerator

    gen = SignalGenerator(nominal_freq=50.0, sample_rate=1200,
                           amplitude_rms=230.0, noise_level=0.0, seed=1)
    t, cycle = gen.generate_cycle()

    phasors = compute_three_phase_phasors(cycle["Va"], cycle["Vb"], cycle["Vc"])

    print("Expected: 230.0 V RMS, angles 0, -120, 120 degrees\n")
    for name, p in phasors.items():
        print(f"{name}: magnitude = {p['magnitude']:.3f} V, angle = {p['angle']:.3f} deg")

    mag1, ang1 = compute_positive_sequence(phasors)
    print(f"\nPositive sequence: magnitude = {mag1:.3f} V, angle = {ang1:.3f} deg")