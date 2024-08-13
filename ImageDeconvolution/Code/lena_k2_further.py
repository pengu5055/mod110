"""
Use Wiener filter and deconvolve the image lena with kernel k1
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import matplotlib as mpl
from skimage import io
import scipy.signal as signal
from skimage import color, data, restoration

# Load the images
lena_k1 = {
    '0': io.imread('./SuppliedData/lena_slike/lena_k2_n0.pgm'),
    '4': io.imread('./SuppliedData/lena_slike/lena_k2_n4.pgm'),
    '8': io.imread('./SuppliedData/lena_slike/lena_k2_n8.pgm'),
    '16': io.imread('./SuppliedData/lena_slike/lena_k2_n16.pgm')
}
kernel1 = io.imread('./SuppliedData/lena_slike/kernel2.pgm')

# Create a 2D window
window_1d = signal.windows.gaussian(kernel1.shape[0], std=120)
window_2d = np.outer(window_1d, window_1d)

deconvolved_images = {}
for noise, image in lena_k1.items():
    deconvolved_images[noise] = restoration.wiener(image * window_2d, kernel1, 1e7)
    deconvolved_images[noise] = deconvolved_images[noise] / np.max(deconvolved_images[noise]) * 255

phases_adjusted = {}
for noise, image in deconvolved_images.items():
    phases_adjusted[noise] = restoration.denoise_tv_bregman(image)
    phases_adjusted[noise] = phases_adjusted[noise] / np.max(phases_adjusted[noise]) * 255


# Display the images
fig, axs = plt.subplots(3, 4, figsize=(14, 8))
for ax in axs.flatten():
    ax.axis('off')

for i, (noise, image) in enumerate(lena_k1.items()):
    axs[0, i].imshow(image, cmap='gray')
    axs[0, i].set_title(f'lena_k2_n{noise}.pgm')

for i, (noise, image) in enumerate(deconvolved_images.items()):
    norm = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    axs[1, i].imshow(image, cmap='gray', norm=norm)
    axs[1, i].set_title(f'Deconvolved')

for i, (noise, image) in enumerate(phases_adjusted.items()):
    norm = mpl.colors.Normalize(vmin=image.min(), vmax=image.max())
    axs[2, i].imshow(image, cmap='gray', norm=norm)
    axs[2, i].set_title(f'Phase Unwrapped')

plt.savefig("./ImageDeconvolution/Images/lena_k2_deconvolved_noperiodic.png", dpi=500)
plt.subplots_adjust(wspace=0.8)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(8, 5))
# [:phases_adjusted['0'].shape[0]//2, :phases_adjusted['0'].shape[1]//2] to trim
spectrum = fft.fft2(phases_adjusted['0'])
norm = mpl.colors.LogNorm(vmin=np.abs(spectrum).min(), vmax=np.abs(spectrum).max())
iron = 35
spectrum[:, iron:-iron] = np.mean(spectrum)
spectrum[iron:-iron, :] = np.mean(spectrum)
corners = [[0, 0], [0, -1], [-1, 0], [-1, -1]]
values = [spectrum[corner[0], corner[1]] for corner in corners]
for corner in corners:
    spectrum[corner[0], corner[1]] = np.mean(spectrum) * 1.5
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
ax[0].imshow(np.abs(spectrum), cmap='gray', norm=norm)
# cbar = plt.colorbar(sm, ax=ax)
ax[0].set_title('Spectrum')

restored = np.abs(fft.ifft2(spectrum))
restored = restored / np.max(restored) * 255
norm = mpl.colors.Normalize(vmin=restored.min(), vmax=restored.max())
ax[1].imshow(restored, cmap='gray', norm=norm)
ax[1].set_title('Restored Image')

ax[2].imshow(phases_adjusted['0'], cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Input to Process')

plt.savefig("./ImageDeconvolution/Images/lena_k2_deconvolved_spectra.png", dpi=500)
plt.tight_layout()
plt.show()
