"""
Plot various windows to display in report introduction.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal

# Use custom style
mpl.style.use('./ma-style.mplstyle')

# Create windows
data_len = 512
windows = {
    'Rectangular': signal.windows.boxcar(data_len),
    'Blackman': signal.windows.blackman(data_len),
    'Hann': signal.windows.hann(data_len),
    'Bartlett': signal.windows.bartlett(data_len),
    'Gaussian': signal.windows.gaussian(data_len, std=100),
    'Hamming': signal.windows.hamming(data_len),
    'Blackman': signal.windows.blackman(data_len),
    'Kaiser': signal.windows.kaiser(data_len, beta=14)
}

colors = ["#37123c","#d72483","#ddc4dd","#60afff","#98CE00"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='compressed')  

plt.suptitle("Window Functions for Signal Processing")
ax[0].plot(np.arange(data_len), windows['Rectangular'], label='Rectangular', color=colors[0])
ax[0].plot(np.arange(data_len), windows['Blackman'], label='Blackman', color=colors[1])
ax[0].plot(np.arange(data_len), windows['Hann'], label='Hann', color=colors[3])
ax[0].plot(np.arange(data_len), windows['Bartlett'], label='Bartlett', color=colors[4])

ax[1].plot(np.arange(data_len), windows['Gaussian'], label='Gaussian', color=colors[0])
ax[1].plot(np.arange(data_len), windows['Hamming'], label='Hamming', color=colors[1])
ax[1].plot(np.arange(data_len), windows['Blackman'], label='Blackman', color=colors[3])
ax[1].plot(np.arange(data_len), windows['Kaiser'], label='Kaiser', color=colors[4])

for i in range(2):
    ax[i].set_xlabel('Sample')
    ax[i].set_ylabel('Amplitude')
    ax[i].legend(frameon=False)

plt.savefig("./SpectralAnalysis/Images/window-functions.png", dpi=500)
plt.show()        
