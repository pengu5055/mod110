"""
Make signal val3.dat periodic to reduce spectral leakage
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
data2 = np.copy(data)
data2[0] = data2[-1]
data3 = np.copy(data)
data3[-10:] = data[-1]

data_len = len(data)
data_time = 1/SAMPLE_RATE * data_len
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)
bare_spectrum = fft.fft(data)
bare_spectrum = bare_spectrum[:data_len//2]
bare_spectrum2 = fft.fft(data2)
bare_spectrum2 = bare_spectrum2[:data_len//2]
bare_spectrum3 = fft.fft(data3)
bare_spectrum3 = bare_spectrum3[:data_len//2]


# Plot the signal
cm = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))
fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='compressed')

ax[0].plot(freq, np.abs(bare_spectrum), color=cm[1], label='Original', alpha=0.8)
ax[0].plot(freq, np.abs(bare_spectrum2), color=cm[4], label='Fixed Periodicity 1-point', alpha=0.8)
ax[0].plot(freq, np.abs(bare_spectrum3), color=cm[7], label='Fixed Periodicity 10-point', alpha=0.8)
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Amplitude [arb. u.]')
ax[0].set_title('Signal val3.dat Spectra')
l = ax[0].legend(frameon=True)
l.set_title('Signal')
f = l.get_frame()
f.set_edgecolor('black')
f.set_facecolor('white')
f.set_linewidth(0.5)
f.set_alpha(0.8)

ax[1].plot(freq, np.abs(np.abs(bare_spectrum) - np.abs(bare_spectrum2)), color=cm[3], label="1-point Fix™")
ax[1].plot(freq, np.abs(np.abs(bare_spectrum) - np.abs(bare_spectrum3)), color=cm[7], label="10-point Fix™")
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude [arb. u.]')
ax[1].set_title('Absolute Difference')
ax[1].set_yscale('log')
l = ax[1].legend(frameon=True)
l.set_title('Performed Fix')
f = l.get_frame()
f.set_edgecolor('black')
f.set_facecolor('white')
f.set_linewidth(0.5)
f.set_alpha(0.8)

plt.suptitle(f"Signal {data_path.split('/')[-1]} w/ Ghetto Periodicity Fix™")
plt.savefig(f"./SpectralAnalysis/Images/fixed-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()
