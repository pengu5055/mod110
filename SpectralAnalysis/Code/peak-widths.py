"""
Calculate and fit the peak widths of the spectral lines using when
using different windows.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import scipy.fft as fft
from analysis import *
import cmasher as cmr

# Constants
SAMPLE_RATE = 44100

# Use custom style
mpl.style.use('./ma-style.mplstyle')

# Load the signal
data_path = './SuppliedData/val2.dat'
data = np.loadtxt(data_path)
data = data / np.max(data)
data_len = len(data)
data_time = 1/SAMPLE_RATE * data_len
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)
bare_spectrum = fft.fft(data)
bare_spectrum = bare_spectrum[:data_len//2]

# Create windows
windows = {
    'Rectangular': signal.windows.boxcar(data_len),
    'Blackman': signal.windows.blackman(data_len),
    'Hann': signal.windows.hann(data_len),
    'Bartlett': signal.windows.bartlett(data_len),
    'Gaussian': signal.windows.gaussian(data_len, std=100),
    'Hamming': signal.windows.hamming(data_len),
    'Lanczos': signal.windows.lanczos(data_len),
    'Kaiser': signal.windows.kaiser(data_len, beta=14)
}

# Create spectra
spectra = {}
for window in windows:
    spectra[window] = fft.fft(data * windows[window])
    spectra[window] = spectra[window][:data_len//2]

# Bare peak dimensions
bare_peaks = signal.find_peaks(np.abs(bare_spectrum), height=30)
bare_peak_widths = signal.peak_widths(np.abs(bare_spectrum), bare_peaks[0], rel_height=0.5)
bare_peak_widths = bare_peak_widths[0] * SAMPLE_RATE / data_len

# Calculate peak widths
peaks = {}
for window in windows:
    peaks[window] = signal.find_peaks(np.abs(spectra[window]), height=10)
    peaks[window] = np.array(peaks[window][1]["peak_heights"]) / bare_peaks[1]["peak_heights"]

# Calculate peak widths -> FWHM
peak_widths = {}
for window in windows:
    peak = signal.find_peaks(np.abs(spectra[window]), height=10)
    peak_widths[window] = signal.peak_widths(np.abs(spectra[window]), peak[0], rel_height=0.1)
    print(peak_widths[window])
    peak_widths[window] = (peak_widths[window][0] * SAMPLE_RATE/data_len) / bare_peak_widths


colors = ["#37123c","#d72483","#ddc4dd","#60afff","#98CE00"]
cm = cmr.take_cmap_colors('cmr.tropical', len(windows), cmap_range=(0.0, 0.85))
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='compressed')
x_axis = np.arange(len(windows))

ax[0].set_title("Relative Peak Heights")
ax[0].bar(x_axis, [np.mean(peaks[window]) for window in windows], color=cm, alpha=1,
          zorder=10, edgecolor='black', linewidth=0.5)
ax[0].fill_between([-0.4, 0.4], 0, 1, color=cm[0], edgecolor="black", linewidth=0.5, hatch="\\\\", zorder=10)
ax[0].set_xticks(x_axis)
ax[0].set_xticklabels(windows.keys(), rotation=30)
# ax[0].set_xlabel("Window Function")
ax[0].set_ylabel("Peak Heights")


ax[1].set_title("Relative Peak Widths")
print(peak_widths)
ax[1].bar(x_axis, [np.mean(peak_widths[window]) for window in windows], color=cm, alpha=1,
          zorder=10, edgecolor='black', linewidth=0.5)
ax[1].set_xticks(x_axis)
ax[1].fill_between([-0.4, 0.4], 0, 1, color=cm[0], edgecolor="black", linewidth=0.5, hatch="\\\\", zorder=10)
ax[1].set_xticklabels(windows.keys(), rotation=30)
# ax[1].set_xlabel("Window Function")
ax[1].set_ylabel("Peak Widths")

plt.suptitle(f"Relative Peak Heights and Widths for Different Windows on {data_path.split('/')[-1]}")
plt.savefig(f"./SpectralAnalysis/Images/peak-widths-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()
