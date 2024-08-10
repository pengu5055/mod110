"""
Determine frequency spectrum of the signal on 512 samples.
Observe what happens with different window functions and reduced sample size.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft

# Load the signal
data = np.loadtxt('./SuppliedData/val3.dat')
data_len = len(data)
han_window = signal.windows.hann(data_len)

data = data / np.max(data)

# Plot the signal
plt.plot(data)
plt.plot(han_window)
plt.title("Signal")

plt.show()
