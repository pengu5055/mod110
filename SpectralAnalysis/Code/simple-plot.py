"""
Plot signals and spectra with out and frills.
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

# Load second signal
data_path2 = './SuppliedData/val3.dat'
data2 = np.loadtxt(data_path2)
data2 = data2 / np.max(data2)
data_len2 = len(data2)
data_time2 = 1/SAMPLE_RATE * data_len2
bare_spectrum2 = fft.fft(data2)
bare_spectrum2 = bare_spectrum2[:data_len2//2]

# Plot
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))
fig = plt.figure(figsize=(12, 8), layout='compressed')
gs = fig.add_gridspec(2,2)
ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])]

# Plot the 1. signal
ax[0].plot(data, color=colors[0], alpha=0.8)
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Amplitude [arb. u.]')
ax[0].set_title(f"Raw Signal: {data_path.split('/')[-1]}")

# Plot the 2. signal
ax[1].plot(data2, color=colors[2], alpha=0.8)
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('Amplitude [arb. u.]')
ax[1].set_title(f"Raw Signal: {data_path2.split('/')[-1]}")

# Plot both spectra
ax[2].plot(freq, np.abs(bare_spectrum), label=f"{data_path.split('/')[-1]}", color=colors[0], alpha=0.8, lw=1.5)
ax[2].plot(freq, np.abs(bare_spectrum2), label=f"{data_path2.split('/')[-1]}", color=colors[2], alpha=0.8, lw=1.5)
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Amplitude [arb. u.]')
ax[2].set_title("Bare Spectra of Both Signals")
l2 = ax[2].legend(frameon=True)
f2 = l2.get_frame()
l2.set_title('Spectra')
f2.set_facecolor('white')
f2.set_edgecolor('black')
f2.set_linewidth(0.5)
f2.set_alpha(0.8)

plt.savefig('./SpectralAnalysis/Images/raw-signals-spectra.png', dpi=500)
plt.show()
