"""
Use Wiener filter and deconvolve the image lena with kernel k1, using different windows
"""
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib as mpl
from skimage import io
import scipy.signal as signal
from skimage import color, data, restoration

# Use custom style 
mpl.style.use('./ma-style.mplstyle')

# Load the images
lena_k1 = io.imread('./SuppliedData/lena_slike/lena_k1_n16.pgm')
kernel1 = io.imread('./SuppliedData/lena_slike/kernel1.pgm')
data_len = len(lena_k1)
exclusion = 20

# Create a 2D window
windows = {
    'Zero Pad': np.pad(signal.windows.boxcar(data_len - exclusion), exclusion//2, mode='constant'),
    'Blackman Window': signal.windows.blackman(data_len),
    'Hann Window': signal.windows.hann(data_len),
    'Bartlett Window': signal.windows.bartlett(data_len),
    'Gaussian Window': signal.windows.gaussian(data_len, std=180),
    'Hamming Window': signal.windows.hamming(data_len),
    'Lanczos Window': signal.windows.lanczos(data_len),
    'Kaiser Window': signal.windows.kaiser(data_len, beta=3)
}

windows_2d = {name: np.outer(window, window) for name, window in windows.items()}

# Deconvolve the images
windowed_images = {}
for flavour, window in windows_2d.items():
    windowed_images[flavour] = lena_k1 * window

deconvolved_images = {}
for flavour, image in windowed_images.items():
    deconvolved_images[flavour] = restoration.wiener(image, kernel1, 1e7)
    deconvolved_images[flavour] = deconvolved_images[flavour] / np.max(deconvolved_images[flavour]) * 255


# Display the images
fig, axs = plt.subplots(3, 3, figsize=(10, 8), layout='compressed')
axs = axs.flatten()
for ax in axs:
    ax.axis('off')

for i, (flavour, image) in enumerate(deconvolved_images.items()):
    idx = i + 1
    norm = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    axs[idx].imshow(image, cmap='gray', norm=norm)
    axs[idx].set_title(f'{flavour}')

axs[0].set_title('Original')
axs[0].imshow(lena_k1, cmap='gray')

plt.savefig("./ImageDeconvolution/Images/lena_k1_n16_windows.png", dpi=500)
plt.show()
