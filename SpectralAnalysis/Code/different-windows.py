"""
Plot signal and spectrum with different window functions.
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
data_path = './SuppliedData/val3.dat'
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

# Plot the signal
colors = ["#37123c","#d72483","#ddc4dd","#60afff","#98CE00"]
cm = cmr.take_cmap_colors('cmr.tropical', len(windows), cmap_range=(0.0, 0.85))
fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout='compressed')

# ax.plot(freq, abs(bare_spectrum), label='Bare', color=colors[0])
for i, spectrum in enumerate(spectra):
     ax.plot(freq, np.abs(np.abs(bare_spectrum) - np.abs(spectra[spectrum])), alpha=0.8,
             color=cm[i], label=spectrum)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude [arb. u.]')
l = ax.legend(frameon=True, ncols=2)
l.set_title('Window Function')
f = l.get_frame()
f.set_edgecolor('black')
f.set_facecolor('white')
f.set_linewidth(0.5)
f.set_alpha(0.8)
ax.set_title("Absolute Difference Between Bare and Windowed Spectra")
ax.set_yscale('log')

plt.suptitle(f"Signal {data_path.split('/')[-1]} Processed with Different Windows")
plt.savefig(f"./SpectralAnalysis/Images/window-spectra-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()
