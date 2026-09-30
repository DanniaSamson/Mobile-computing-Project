import numpy as np
import matplotlib.pyplot as plt
from signal_generator import SignalGenerator

# Create the signal generator
gen = SignalGenerator(nominal_freq=50.0, sample_rate=1200,
                      amplitude_rms=230.0, noise_level=0.0, seed=1)

# Generate 3 smooth cycles for the combined top graph
t_smooth, signals_smooth = gen.generate_samples(0.06)

# Get one cycle sample points for the dots
gen2 = SignalGenerator(nominal_freq=50.0, sample_rate=1200,
                       amplitude_rms=230.0, noise_level=0.0, seed=1)
t_cycle, cycle = gen2.generate_cycle()

# ---- PRINT THE VALUES ----
print("=" * 55)
print("   SIGNAL GENERATOR OUTPUT - ONE CYCLE (24 SAMPLES)")
print("=" * 55)
print(f"\n{'Sample':<10} {'Va (V)':<15} {'Vb (V)':<15} {'Vc (V)':<15}")
print("-" * 55)
for i in range(len(t_cycle)):
    print(f"{i:<10} {cycle['Va'][i]:<15.2f} "
          f"{cycle['Vb'][i]:<15.2f} {cycle['Vc'][i]:<15.2f}")
print("\n" + "=" * 55)
print(f"Peak voltage  = {max(cycle['Va']):.2f} V")
print(f"Bottom value  = {min(cycle['Va']):.2f} V")
print(f"Samples/cycle = {len(cycle['Va'])}")
print("=" * 55)

# ---- DRAW THE GRAPH ----
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
fig.suptitle("3-Phase Voltage Signal Generator Output",
             fontsize=14, fontweight='bold')

# ---- TOP GRAPH — smooth combined 3 cycles ----
axes[0].set_title("Combined Output — All 3 Phases Together (3 cycles)",
                  fontsize=11, fontweight='bold')
axes[0].plot(t_smooth * 1000, signals_smooth['Va'],
             color='tab:red',   linewidth=2, label='Phase A (0°)')
axes[0].plot(t_smooth * 1000, signals_smooth['Vb'],
             color='tab:green', linewidth=2, label='Phase B (-120°)')
axes[0].plot(t_smooth * 1000, signals_smooth['Vc'],
             color='tab:blue',  linewidth=2, label='Phase C (+120°)')
axes[0].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
axes[0].set_ylabel("Voltage (V)")
axes[0].set_ylim(-380, 380)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.01, 0.95,
             "This is the generated 3-phase voltage signal",
             transform=axes[0].transAxes,
             fontsize=9, color='darkblue',
             verticalalignment='top')
axes[0].set_xlabel("Time (milliseconds)")

# ---- BOTTOM 3 GRAPHS — each phase separately ----
phases = [
    ("Va", "tab:red",   "Phase A — starts at peak (0°)"),
    ("Vb", "tab:green", "Phase B — starts at -120°"),
    ("Vc", "tab:blue",  "Phase C — starts at +120°"),
]

for ax, (name, color, title) in zip(axes[1:], phases):
    ax.set_title(title, fontsize=10, color=color)

    # smooth continuous curve — 3 cycles
    t_s, sig = SignalGenerator(noise_level=0.0,
                               seed=1).generate_samples(0.06)
    ax.plot(t_s * 1000, sig[name],
            color=color, linewidth=2, label=f'{name} smooth curve')

    # 24 sample dots on first cycle only
    gen_dot = SignalGenerator(noise_level=0.0, seed=1)
    t_dot, dot_cycle = gen_dot.generate_cycle()
    ax.scatter(t_dot * 1000, dot_cycle[name],
               color='black', s=40, zorder=5,
               label='24 sample points')

    # sample number labels on each dot
    for i in range(len(t_dot)):
        ax.annotate(f'{i}',
                    xy=(t_dot[i] * 1000, dot_cycle[name][i]),
                    xytext=(0, 8),
                    textcoords='offset points',
                    fontsize=6, color='black',
                    ha='center')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axhline(y=325.27,  color='gray', linewidth=0.5,
               linestyle=':', label='±peak 325.27V')
    ax.axhline(y=-325.27, color='gray', linewidth=0.5, linestyle=':')
    ax.set_ylabel("Voltage (V)")
    ax.set_ylim(-420, 420)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (milliseconds)")

    # shade positive and negative halves
    ax.fill_between(t_s * 1000, sig[name], 0,
                    where=(sig[name] >= 0),
                    alpha=0.08, color='green')
    ax.fill_between(t_s * 1000, sig[name], 0,
                    where=(sig[name] < 0),
                    alpha=0.08, color='red')

    # orange lines at zero crossings
    for i in range(len(sig[name]) - 1):
        if sig[name][i] * sig[name][i+1] < 0:
            ax.axvline(x=t_s[i] * 1000, color='orange',
                       linewidth=1, linestyle='--', alpha=0.6)

    # label peak and bottom with arrows
    peak_idx = int(np.argmax(dot_cycle[name]))
    bottom_idx = int(np.argmin(dot_cycle[name]))
    ax.annotate(f'peak\n{dot_cycle[name][peak_idx]:.1f}V',
                xy=(t_dot[peak_idx]*1000, dot_cycle[name][peak_idx]),
                xytext=(t_dot[peak_idx]*1000 + 2, 300),
                fontsize=7.5, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'bottom\n{dot_cycle[name][bottom_idx]:.1f}V',
                xy=(t_dot[bottom_idx]*1000, dot_cycle[name][bottom_idx]),
                xytext=(t_dot[bottom_idx]*1000 + 2, -300),
                fontsize=7.5, color='darkred',
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.show()