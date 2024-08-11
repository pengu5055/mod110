"""
Plot various windows to display in report introduction.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import cmasher as cmr

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
    'Lanczos': signal.windows.lanczos(data_len),
    'Kaiser': signal.windows.kaiser(data_len, beta=14)
}

colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))
fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout='compressed')  

plt.suptitle("Window Functions for Signal Processing")

ax.plot(windows['Rectangular'], label='Rectangular', color=colors[0])
ax.plot(windows['Blackman'], label='Blackman', color=colors[1])
ax.plot(windows['Hann'], label='Hann', color=colors[2])
ax.plot(windows['Bartlett'], label='Bartlett', color=colors[3])
ax.plot(windows['Gaussian'], label='Gaussian', color=colors[4])
ax.plot(windows['Hamming'], label='Hamming', color=colors[5])
ax.plot(windows['Lanczos'], label='Lanczos', color=colors[6])
ax.plot(windows['Kaiser'], label='Kaiser', color=colors[7])

ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
l = ax.legend(ncols=2)
l.set_title('Window Function')
f = l.get_frame()
f.set_edgecolor('black')
f.set_facecolor('white')
f.set_linewidth(0.5)
f.set_alpha(0.8)
ax.set_xlim(0, data_len)
plt.savefig("./SpectralAnalysis/Images/window-functions.png", dpi=500)
plt.show()        
