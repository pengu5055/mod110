"""
Make Cook AI annoncement trashy by adding noise to it.
"""
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy.io import wavfile

def load_wav(filename):
    fs_rate, signal = wavfile.read(filename)
    # Average stereo channels
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2
    # Read number of samples, duration and sampling frequency
    n = signal.shape[0]
    secs = n / fs_rate
    T = 1/fs_rate
    t = np.arange(0, secs, T)
    data = {
        'fs_rate': fs_rate,
        'signal': signal,
        'n': n,
        'secs': secs,
        'T': T,
        't': t
    }

    return data


# Constants
SAMPLE_RATE = 44100



# Load signals
wav = load_wav('./GeneratedData/hood_cook.wav')
data_len = wav["n"]
data_time = wav["secs"]
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)

noise = np.random.normal(0, np.max(wav["signal"])//4, data_len)
noisy = wav["signal"] + noise
noisy = noisy / np.max(noisy)

# Save the noisy signal
wavfile.write('./GeneratedData/hood_cook_noisy.wav', wav["fs_rate"], noisy)
