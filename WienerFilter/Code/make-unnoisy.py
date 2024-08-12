"""
Make Cook AI annoncement GREAT AGAIN! by removing noise from it.
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

# Comb filter parameters
delay_times = np.linspace(0.05, 100.05, 100)
feedback_gains = np.linspace(0.9, 0.1, 100)

reverb_signals = []
max_delay_samples = int(max(delay_times) * wav['fs_rate'])

# Pad the signal with zeros
wav["signal"] = np.pad(wav["signal"], (0, max_delay_samples), 'constant')

# Apply comb filter to each copy of the signal
for i in range(len(delay_times)):
    delay_samples = int(delay_times[i] * wav['fs_rate'])
    delayed_signal = np.concatenate((np.zeros(delay_samples), wav["signal"][:-delay_samples]))
    padded_signal = np.pad(wav["signal"], (0, len(delayed_signal) - len(wav["signal"])), 'constant', constant_values=(0, wav["signal"][-1]))
    feedback_signal = feedback_gains[i] * padded_signal
    b = signal.firwin(100, cutoff=5000, fs=wav["fs_rate"])
    filtered_signal = signal.lfilter(b, 1, feedback_signal)
    reverb_signal = padded_signal + feedback_signal
    reverb_signals.append(reverb_signal)


# Save the reverb signals
reverb_sum = np.sum(reverb_signals, axis=0)
sound_out = reverb_sum / np.max(reverb_sum)

wavfile.write('./GeneratedData/hood_cook_ambient.wav', wav["fs_rate"], sound_out)
