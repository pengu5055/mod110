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
print(f"Data length: {data_len}, Time: {data_time}")
spectrum = fft.fft(data)
spectrum = spectrum[:data_len//2]
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)
peaks = signal.find_peaks(np.abs(spectrum), height=30)
print(f"Peaks: {[freq[p] for p in peaks[0]]}")
print(f"Peak Heights: {np.array(peaks[1]['peak_heights'])/np.max(np.abs(spectrum))}")

# Load .wav 
sound_data = load_wav('./GeneratedData/sig_4323_10463_12971_18245.wav')
print(sound_data["n"])
sound = sound_data['signal'][:data_len] / np.max(sound_data['signal'])
s_spectrum = fft.fft(sound)
s_spectrum = s_spectrum[:data_len//2]


# Plot the signal
colors = ["#37123c","#d72483","#ddc4dd","#60afff","#98CE00"]
fig, ax = plt.subplots(3, 1, figsize=(10, 8), layout='compressed')

ax[0].plot(data, label='Signal', color=colors[0])
ax[0].plot(sound, label='Recreation', color=colors[1], alpha=0.5)
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Amplitude [arb. u.]')
ax[0].legend(frameon=True)

ax[1].plot(freq, np.abs(spectrum), label='Real', color=colors[3])
ax[1].plot(freq, np.abs(s_spectrum), label='Recreation', color=colors[4])
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude [arb. u.]')
ax[1].legend(frameon=False)

ax[2].plot(np.abs(data-sound), label='Difference', color=colors[1])
ax[2].set_xlabel('Sample')
ax[2].set_ylabel('Absolute Difference')
ax[2].legend(frameon=False)
ax[2].set_yscale('log')

plt.suptitle("Input Signal Reconstruction")

plt.savefig(f"./SpectralAnalysis/Images/reconstruction-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()        
