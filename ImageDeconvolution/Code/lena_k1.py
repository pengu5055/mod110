"""
Use Wiener filter and deconvolve the image lena with kernel k1
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
lena_k1 = {
    '0': io.imread('./SuppliedData/lena_slike/lena_k1_n0.pgm'),
    '4': io.imread('./SuppliedData/lena_slike/lena_k1_n4.pgm'),
    '8': io.imread('./SuppliedData/lena_slike/lena_k1_n8.pgm'),
    '16': io.imread('./SuppliedData/lena_slike/lena_k1_n16.pgm')
}
kernel1 = io.imread('./SuppliedData/lena_slike/kernel1.pgm')

# Create a 2D window
window_1d = signal.windows.hann(int(kernel1.shape[0] - 0.1*kernel1.shape[0]))
window_1d = np.pad(window_1d, kernel1.shape[0] // 2 - len(window_1d) // 2, mode='constant')
window_2d = np.outer(window_1d, window_1d)

window2_1d = signal.windows.kaiser(kernel1.shape[0], beta=5)
window2_2d = np.outer(window2_1d, window2_1d)

# Deconvolve the images
windowed_images = {}
for noise, image in lena_k1.items():
    windowed_images[noise] = image * window_2d

deconvolved_images = {}
for noise, image in lena_k1.items():
    deconvolved_images[noise] = restoration.wiener(image, kernel1, 1e6)
    deconvolved_images[noise] = deconvolved_images[noise] / np.max(deconvolved_images[noise]) * 255

windowed_deconvolved = {}
for noise, image in windowed_images.items():
    windowed_deconvolved[noise] = restoration.wiener(image, kernel1, 1e6)
    windowed_deconvolved[noise] = windowed_deconvolved[noise] / np.max(windowed_deconvolved[noise]) * 255

windowed2_deconvolved = {}
for noise, image in lena_k1.items():
    windowed2_deconvolved[noise] = restoration.wiener(image * window2_2d, kernel1, 1e6)
    windowed2_deconvolved[noise] = windowed2_deconvolved[noise] / np.max(windowed2_deconvolved[noise]) * 255

# Display the images
fig, axs = plt.subplots(3, 4, figsize=(14, 8))
for ax in axs.flatten():
    ax.axis('off')

for i, (noise, image) in enumerate(lena_k1.items()):
    axs[0, i].imshow(image, cmap='gray')
    axs[0, i].set_title(f'lena_k1_n{noise}.pgm')

for i, (noise, image) in enumerate(deconvolved_images.items()):
    norm = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    axs[1, i].imshow(image, cmap='gray', norm=norm)
    axs[1, i].set_title(f'Deconvolved')

for i, (noise, image) in enumerate(windowed_deconvolved.items()):
    norm = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    axs[2, i].imshow(image, cmap='gray', norm=norm)
    axs[2, i].set_title(f'w/ Hann Window')


plt.savefig("./ImageDeconvolution/Images/lena_k1_deconvolved.png", dpi=500)
plt.subplots_adjust(wspace=0.8)
plt.tight_layout()
plt.show()
