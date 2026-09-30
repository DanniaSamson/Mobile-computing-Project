"""
signal_generator.py
--------------------
Creates a fake (synthetic) 3-phase voltage signal.

A real PMU samples voltage/current from CTs and PTs on the power grid.
For this project we simulate that with three balanced sinusoids,
120 degrees apart (phases A, B, C), with optional noise and a slow
amplitude "wobble" so the output looks a bit more realistic.

Outputs: sine wave sample arrays (numpy arrays) for Va, Vb, Vc.
"""

import numpy as np


class SignalGenerator:
    def __init__(self, nominal_freq=50.0, sample_rate=1200,
                 amplitude_rms=230.0, noise_level=0.01, seed=None):
        """
        nominal_freq   : grid frequency in Hz (50 or 60)
        sample_rate    : samples per second. Must be an integer multiple
                          of nominal_freq so each "cycle" has a whole
                          number of samples (needed for the DFT phasor
                          calculation later).
        amplitude_rms  : nominal RMS voltage of each phase (volts)
        noise_level    : fraction of amplitude added as random noise
                          (0.0 = perfectly clean signal)
        seed           : optional RNG seed for reproducible noise
        """
        if sample_rate % nominal_freq != 0:
            raise ValueError(
                "sample_rate must be a whole multiple of nominal_freq "
                "so every cycle has the same number of samples."
            )

        self.nominal_freq = nominal_freq
        self.sample_rate = sample_rate
        self.amplitude_rms = amplitude_rms
        self.amplitude_peak = amplitude_rms * np.sqrt(2)
        self.noise_level = noise_level
        self.samples_per_cycle = int(sample_rate / nominal_freq)
        self._rng = np.random.default_rng(seed)

        # Standard ABC phase rotation, 120 degrees apart
        self.phase_offsets_deg = {"Va": 0.0, "Vb": -120.0, "Vc": 120.0}

        # global sample counter so consecutive cycles have continuous phase
        self._n0 = 0

    def generate_cycle(self):
        """
        Generate exactly one electrical cycle worth of samples for all
        three phases. Returns a dict: {"Va": np.array, "Vb": np.array,
        "Vc": np.array} each of length `samples_per_cycle`, plus the
        time axis for that cycle.

        Calling this repeatedly produces a continuous, phase-coherent
        waveform (no jumps between cycles).
        """
        n = np.arange(self._n0, self._n0 + self.samples_per_cycle)
        t = n / self.sample_rate
        w = 2 * np.pi * self.nominal_freq

        signals = {}
        for name, phase_deg in self.phase_offsets_deg.items():
            phase_rad = np.radians(phase_deg)
            clean = self.amplitude_peak * np.cos(w * t + phase_rad)
            if self.noise_level > 0:
                noise = self._rng.normal(
                    0, self.noise_level * self.amplitude_peak, size=clean.shape
                )
                clean = clean + noise
            signals[name] = clean

        self._n0 += self.samples_per_cycle
        return t, signals

    def generate_samples(self, duration_sec):
        """
        Convenience/batch method (mainly for testing & the visualizer's
        offline demo mode): generate `duration_sec` worth of samples in
        one call instead of cycle-by-cycle.

        Returns (t, signals) where signals = {"Va":..., "Vb":..., "Vc":...}
        """
        total_samples = int(duration_sec * self.sample_rate)
        n = np.arange(total_samples)
        t = n / self.sample_rate
        w = 2 * np.pi * self.nominal_freq

        signals = {}
        for name, phase_deg in self.phase_offsets_deg.items():
            phase_rad = np.radians(phase_deg)
            clean = self.amplitude_peak * np.cos(w * t + phase_rad)
            if self.noise_level > 0:
                noise = self._rng.normal(
                    0, self.noise_level * self.amplitude_peak, size=clean.shape
                )
                clean = clean + noise
            signals[name] = clean

        return t, signals


if __name__ == "__main__":
    # Quick self-test / demo
    gen = SignalGenerator(nominal_freq=50.0, sample_rate=1200,
                           amplitude_rms=230.0, noise_level=0.0, seed=1)

    print(f"samples_per_cycle = {gen.samples_per_cycle}")
    t, cycle1 = gen.generate_cycle()
    print("First cycle, Va samples:")
    print(np.round(cycle1["Va"], 2))

    t2, cycle2 = gen.generate_cycle()
    print("\nSecond cycle starts where the first left off (phase-continuous):")
    print(f"last sample of cycle1 Va = {cycle1['Va'][-1]:.3f}")
    print(f"first sample of cycle2 Va = {cycle2['Va'][0]:.3f} "
          f"(should look like the natural next point on the sine wave)")