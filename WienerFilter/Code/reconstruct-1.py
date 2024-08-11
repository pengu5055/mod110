"""
Reconstruct the signal signal1.dat using the Wiener filter.
The expected output is signal0.dat
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import scipy.fft as fft
import cmasher as cmr

# Constants
SAMPLE_RATE = 44100

# Use custom style
mpl.style.use('./ma-style.mplstyle')

# Load signals
signal0 = np.loadtxt('./SuppliedData/signal0.dat')
signal0 = signal0 / np.max(signal0)
signal1 = np.loadtxt('./SuppliedData/signal1.dat')
signal1 = signal1 / np.max(signal1)
data_len = len(signal0)
data_time = 1/SAMPLE_RATE * data_len
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)

# Compute the Wiener filter
filtered = {
    '5': signal.wiener(signal1, mysize=5),
    '15': signal.wiener(signal1, mysize=15),
    '25': signal.wiener(signal1, mysize=25),
    '55': signal.wiener(signal1, mysize=55),
    '85': signal.wiener(signal1, mysize=85),
    '125': signal.wiener(signal1, mysize=125),
    '155': signal.wiener(signal1, mysize=155),
    '185': signal.wiener(signal1, mysize=185),
}

# Plot the signals
colors = cmr.take_cmap_colors('cmr.tropical', len(filtered), cmap_range=(0.0, 0.85))
fig = plt.figure(figsize=(12, 10))
gs_outer = fig.add_gridspec(2, 1, height_ratios=[1, 8])
gs = mpl.gridspec.GridSpecFromSubplotSpec(8, 3, subplot_spec=gs_outer[1],
                      hspace=0.1, wspace=0.05)
gs_upper = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0])
ax_u = [
    fig.add_subplot(gs_upper[0, 0]),
    fig.add_subplot(gs_upper[0, 1]),
    fig.add_subplot(gs_upper[0, 2]),
]
# Remove the [0, 0] and [0, 2] axes
ax_u[0].set_visible(False)
ax_u[2].set_visible(False)
ax = []
for i in range(0, 8):
    for j in range(3):
        ax.append(fig.add_subplot(gs[i, j]))

for i, (key, value) in enumerate(filtered.items()):
    idx = 3*i
    ax[idx].sharex(ax[0])
    ax[idx].plot(value, color=colors[i])
    plt.setp(ax[idx].get_xticklabels(), visible=False)
    plt.setp(ax[idx].get_xticklines() , visible=False)

for i, (key, value) in enumerate(filtered.items()):
    idx = 3*i+1
    ax[idx].sharex(ax[1])
    ax[idx].sharey(ax[idx-1])
    ax[idx].plot(np.abs(np.abs(signal0) - np.abs(value)), color=colors[i])
    plt.setp(ax[idx].get_xticklabels(), visible=False)
    plt.setp(ax[idx].get_xticklines() , visible=False)
    plt.setp(ax[idx].get_yticklabels(), visible=False)
    plt.setp(ax[idx].get_yticklines() , visible=False)


for i, (key, value) in enumerate(filtered.items()):
    idx = 3*i+2
    ax[idx].sharex(ax[2])
    ax[idx].plot(freq, np.abs(fft.fft(value)[:len(value)//2]), color=colors[i])
    ax[idx].yaxis.tick_right()
    plt.setp(ax[idx].get_xticklabels(), visible=False)
    plt.setp(ax[idx].get_xticklines() , visible=False)

plt.setp(ax[21].get_xticklabels(), visible=True)
plt.setp(ax[22].get_xticklabels(), visible=True)
plt.setp(ax[23].get_xticklabels(), visible=True)
plt.setp(ax[21].get_xticklines(), visible=True)
plt.setp(ax[22].get_xticklines(), visible=True)
plt.setp(ax[23].get_xticklines(), visible=True)
ax[0].set_title('Wiener Filtered Signals')
ax[1].set_title('Abs. Diff. b/w signal0.dat')
ax[2].set_title('Spectrum')
ax[21].set_xticklabels(np.arange(0, data_len, 100))
ax[22].set_xticklabels(np.arange(0, data_len, 100))
ax[23].set_xticks(np.arange(0, freq.max(), 5000))
ax[23].set_xticklabels(np.arange(0, freq.max(), 5000))
ax[23].set_xscale('log')
ax[21].set_xlabel('Samples')
ax[22].set_xlabel('Samples')
ax[23].set_xlabel('Frequency (Hz)')

ax_u[1].plot(signal0, color="black")
ax_u[1].set_title('Noiseless Signal')

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
plt.show()
