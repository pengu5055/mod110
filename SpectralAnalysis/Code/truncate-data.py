"""
Plot signal and spectrum with different window functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import scipy.fft as fft
from analysis import *
import cmasher as cmr
from scipy.interpolate import interp1d

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

# Different number of samples
n_samples = {
    '512': data,
    '256': np.pad(data[:256], (128, 128), mode='constant'),
    '128': np.pad(data[:128], (192, 192), mode='constant'),
    '64': np.pad(data[:64], (224, 224), mode='constant'),
    '32': np.pad(data[:32], (240, 240), mode='constant'),
    '16': np.pad(data[:16], (248, 248), mode='constant'),
}

# Create spectra
spectra = {}
for sample in n_samples:
    spectra[sample] = fft.fft(n_samples[sample])
    spectra[sample] = spectra[sample][:data_len//2]

# interpolated = {}
# for sample in n_samples:
#     interpolated[sample] = np.interp(np.arange(data_len//2), np.arange(len(spectra[sample])), spectra[sample])
#     interpolated[sample] *= np.max(np.abs(spectra[sample])) / np.max(np.abs(interpolated[sample]))

# Get difference between full and reduced samples
# Interpolate the reduced samples to the full length
diff = {}
for sample in n_samples:
    # diff[sample] = np.abs(np.abs(spectra["512"]) - np.abs(interpolated[sample]))
    diff[sample] = np.abs(np.abs(spectra["512"]) - np.abs(spectra[sample]))


# Plot the signal
colors = ["#37123c","#d72483","#ddc4dd","#60afff","#98CE00"]
cm = cmr.take_cmap_colors('cmr.tropical', len(n_samples), cmap_range=(0.0, 0.85))
fig = plt.figure(figsize=(12, 8), layout='compressed')

gs = fig.add_gridspec(2,2)
ax = [fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[0, 1]),
      fig.add_subplot(gs[1, :])]

ax[0].set_title("Absolute Difference")
for i, sample in enumerate(diff):
     ax[0].plot(freq, diff[sample], alpha=0.8, color=cm[i], label=f"Samples: {sample}")

ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Absolute Difference')
ax[0].set_yscale('log')
l1 = ax[0].legend(frameon=True, ncols=2)
l1.set_title('Sample Reduction')
f1 = l1.get_frame()
f1.set_edgecolor('black')
f1.set_facecolor('white')
f1.set_linewidth(0.5)
f1.set_alpha(0.5)


ax[1].set_title("Average Relative Difference")
ax[1].bar(np.arange(len(n_samples)), [np.mean(diff[sample] / np.abs(spectra['512'])) for sample in n_samples], color=cm, alpha=1,
          zorder=10, edgecolor='black', linewidth=0.5)
ax[1].set_xticks(np.arange(len(n_samples)))
ax[1].set_xticklabels([f"{sample}" for sample in n_samples], rotation=0)
ax[1].set_xlabel('Number of Samples')
ax[1].set_ylabel('Average Relative Difference')
ax[1].set_ylim(0.9, 2.2)
ax[1].fill_between([-0.5, 0.5], 0.9, 1.0, color=cm[0], alpha=0.5, linewidth=0.5, edgecolor='black')

ax[2].set_title("Spectra of Zero-padded Signal to Full Length")
for i, sample in enumerate(spectra):
    ax[2].plot(freq, np.abs(spectra[sample]), alpha=0.8, color=cm[i], label=f"Samples: {sample}")
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Amplitude [arb. u.]')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_xlim(1000, np.max(freq))
l3 = ax[2].legend(frameon=True, ncols=2)
l3.set_title('Sample Reduction')
f3 = l3.get_frame()
f3.set_edgecolor('black')
f3.set_facecolor('white')
f3.set_linewidth(0.5)
f3.set_alpha(0.8)


plt.suptitle(f"Signal {data_path.split('/')[-1]} Truncated to Different Number of Samples")
plt.savefig(f"./SpectralAnalysis/Images/reduced-samples-{data_path.split('/')[-1]}.png", dpi=500)
plt.show()
