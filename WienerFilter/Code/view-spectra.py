"""
Load signals and view their spectra.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import scipy.signal as signal
import scipy.fft as fft

# Constants
SAMPLE_RATE = 44100

# Use custom style
mpl.style.use('./ma-style.mplstyle')

# Load signals
signal0 = np.loadtxt('./SuppliedData/signal0.dat')
signal0 = signal0 / np.max(signal0)
signal1 = np.loadtxt('./SuppliedData/signal1.dat')
signal1 = signal1 / np.max(signal1)
signal2 = np.loadtxt('./SuppliedData/signal2.dat')
signal2 = signal2 / np.max(signal2)
signal3 = np.loadtxt('./SuppliedData/signal3.dat')
signal3 = signal3 / np.max(signal3)

spectrum0 = fft.fft(signal0)
spectrum0 = spectrum0[:len(signal0)//2]
dcless_spectrum0 = spectrum0
dcless_spectrum0[0] = 0
spectrum1 = fft.fft(signal1)
spectrum1 = spectrum1[:len(signal1)//2]
dcless_spectrum1 = spectrum1
dcless_spectrum1[0] = 0
spectrum2 = fft.fft(signal2)
spectrum2 = spectrum2[:len(signal2)//2]
dcless_spectrum2 = spectrum2
dcless_spectrum2[0] = 0
spectrum3 = fft.fft(signal3)
spectrum3 = spectrum3[:len(signal3)//2]
dcless_spectrum3 = spectrum3
dcless_spectrum3[0] = 0



data_len = len(signal0)
data_time = data_len / SAMPLE_RATE
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)
sample_range = np.arange(data_len)

# Plot the signals
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='compressed')
ax = axes.flatten()

ax[0].plot(freq, np.abs(spectrum0), color=colors[0])
ax[0].set_title('signal0.dat')
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('Amplitude [arb. u.]')
ax[0].set_xscale('log')

ax[1].plot(freq, np.abs(spectrum1), color=colors[2])
ax[1].set_title('signal1.dat')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Amplitude [arb. u.]')
ax[1].set_xscale('log')

ax[2].plot(freq, np.abs(spectrum2), color=colors[4])
ax[2].set_title('signal2.dat')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('Amplitude [arb. u.]')
ax[2].set_xscale('log')

ax[3].plot(freq, np.abs(spectrum3), color=colors[6])
ax[3].set_title('signal3.dat')
ax[3].set_xlabel('Frequency [Hz]')
ax[3].set_ylabel('Amplitude [arb. u.]')
ax[3].set_xscale('log')
plt.suptitle('Spectra of Signals')
plt.savefig("./WienerFilter/Images/signals-spectra.png", dpi=500)
plt.show()
