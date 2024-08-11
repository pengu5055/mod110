"""
Determine frequency spectrum of the signal on 512 samples.
Observe what happens with different window functions and reduced sample size.
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
print(f"Data length: {data_len}, Time: {data_time}")
spectrum = fft.fft(data)
spectrum = spectrum[:data_len//2]
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)
peaks = signal.find_peaks(np.abs(spectrum), height=30)
print(f"Peaks: {[freq[p] for p in peaks[0]]}")
print(f"Peak Heights: {np.array(peaks[1]['peak_heights'])/np.max(np.abs(spectrum))}")

# Load .wav 
# sound_data = load_wav('./GeneratedData/sig_4323_10463_12971_18245.wav')
sound_data = load_wav('./GeneratedData/sig_5880_9685.wav')
print(sound_data["n"])
sound = sound_data['signal'][:data_len] / np.max(sound_data['signal'])
s_spectrum = fft.fft(sound)
s_spectrum = s_spectrum[:data_len//2]


# Plot the signal
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))

fig, ax = plt.subplots(3, 1, figsize=(10, 8), layout='compressed')

ax[0].plot(data, label='Signal', color=colors[2])
ax[0].plot(sound, label='Recreation', color=colors[0], alpha=0.5)
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Amplitude [arb. u.]')
ax[0].set_title(f"{data_path.split('/')[-1]} and its Recreation")
l0 = ax[0].legend(frameon=True)
l0.set_title('Signal')
f0 = l0.get_frame()
f0.set_edgecolor('black')
f0.set_facecolor('white')
f0.set_linewidth(0.5)
f0.set_alpha(0.8)

ax[1].plot(freq, np.abs(spectrum), label='Real', color=colors[2])
ax[1].plot(freq, np.abs(s_spectrum), label='Recreation', color=colors[0])
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude [arb. u.]')
ax[1].set_title('Spectrum of the Signals')
l1 = ax[1].legend(frameon=True)
l1.set_title('Signal')
f1 = l1.get_frame()
f1.set_edgecolor('black')
f1.set_facecolor('white')
f1.set_linewidth(0.5)
f1.set_alpha(0.8)

ax[2].plot(np.abs(data-sound), label='Difference', color=colors[6])
ax[2].set_xlabel('Sample')
ax[2].set_ylabel('Absolute Difference')
ax[2].set_yscale('log')
ax[2].set_title('Absolute Difference between Signals')

plt.suptitle("Input Signal Reconstruction")

plt.savefig(f"./SpectralAnalysis/Images/reconstruction-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()        
