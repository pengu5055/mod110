"""
Restore the images with periodic perturbations added. *_nx.pgm
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import matplotlib as mpl
from skimage import io
import scipy.signal as signal
import cmasher as cmr
from skimage import color, data, restoration, filters

# Use custom style 
mpl.style.use('./ma-style.mplstyle')

# Load the images
lena_k1 = io.imread('./SuppliedData/lena_slike/lena_k1_nx.pgm')
kernel1 = io.imread('./SuppliedData/lena_slike/kernel1.pgm')
lena_k2 = io.imread('./SuppliedData/lena_slike/lena_k2_nx.pgm')
kernel2 = io.imread('./SuppliedData/lena_slike/kernel2.pgm')
lena_k3 = io.imread('./SuppliedData/lena_slike/lena_k3_nx.pgm')
kernel3 = io.imread('./SuppliedData/lena_slike/kernel3.pgm')

# Create a 2D window
window1_1d = signal.windows.gaussian(kernel1.shape[0], std=180)
window1_2d = np.outer(window1_1d, window1_1d)
window2_1d = signal.windows.gaussian(kernel2.shape[0], std=170)
window2_2d = np.outer(window2_1d, window2_1d)
window3_1d = signal.windows.gaussian(kernel3.shape[0], std=200)
window3_2d = np.outer(window3_1d, window3_1d)

sklearn_k1 = restoration.wiener(lena_k1 * window1_2d, kernel1, 1e7)
sklearn_k1 = filters.butterworth(sklearn_k1, 0.045, high_pass=False)
sklearn_k1 = restoration.denoise_tv_bregman(sklearn_k1)
sklearn_k1 /= np.max(sklearn_k1) * 255

sklearn_k2 = restoration.wiener(lena_k2 * window2_2d, kernel2, 1e8)
sklearn_k2 = filters.butterworth(sklearn_k2, 0.035, high_pass=False)
sklearn_k2 /= np.max(sklearn_k2) * 255

sklearn_k3 = restoration.denoise_tv_bregman(lena_k3)
sklearn_k3 = filters.butterworth(sklearn_k3, 0.35, high_pass=False)
sklearn_k3 = restoration.wiener(sklearn_k3*window3_2d, kernel3, 1e8)
sklearn_k3 = restoration.denoise_tv_bregman(sklearn_k3)
sklearn_k3 = filters.unsharp_mask(sklearn_k3, radius=5, amount=2)
sklearn_k3 /= np.max(sklearn_k3) * 255

# Display the images
fig, ax = plt.subplots(3, 3, figsize=(10, 8), layout='compressed')
for a in ax.flatten():
    a.axis('off')

ax[0, 0].imshow(lena_k1, cmap='gray')
ax[0, 0].set_title('lena_k1_nx.pgm')
ax[0, 1].imshow(lena_k2, cmap='gray')
ax[0, 1].set_title('lena_k2_nx.pgm')
ax[0, 2].imshow(lena_k3, cmap='gray')
ax[0, 2].set_title('lena_k3_nx.pgm')

# Add text boxes that explain processing
bbox = dict(edgecolor="black", linewidth=0.5, facecolor='white', alpha=1)
text1 = "Gaussian Window $\\sigma=180$\nWiener Deconvolution\nButterworth Filter\nTV Bregman Denoising"
text2 = "Gaussian Window $\\sigma=170$\nWiener Deconvolution\nButterworth Filter"
text3 = "TV Bregman Denoising\nButterworth Filter\nGaussian Window $\\sigma=200$\nWiener Deconvolution\nTV Bregman Denoising\nUnsharp Masking"
ax[1, 0].text(0.5, 0.5, text1, horizontalalignment='center',
                verticalalignment='center', transform=ax[1, 0].transAxes, bbox=bbox, fontsize=10, zorder=10)
ax[1, 1].text(0.5, 0.5, text2, horizontalalignment='center',
                verticalalignment='center', transform=ax[1, 1].transAxes, bbox=bbox, fontsize=10, zorder=10)
ax[1, 2].text(0.5, 0.5, text3, horizontalalignment='center',
                verticalalignment='center', transform=ax[1, 2].transAxes, bbox=bbox, fontsize=10, zorder=10)

# Add arrows going from the images to the text boxes
arrowprops = dict(arrowstyle="->", color='black', lw=1.5)
ax[1, 0].plot([0, 1], [0, 1], alpha=0)
ax[1, 0].annotate("", xy=(0.5, 0.), xytext=(0.5, 1.0), arrowprops=arrowprops, zorder=1)
ax[1, 1].plot([0, 1], [0, 1], alpha=0)
ax[1, 1].annotate("", xy=(0.5, 0.), xytext=(0.5, 1.0), arrowprops=arrowprops, zorder=1)
ax[1, 2].plot([0, 1], [0, 1], alpha=0)
ax[1, 2].annotate("", xy=(0.5, 0.), xytext=(0.5, 1.0), arrowprops=arrowprops, zorder=1)

ax[2, 0].imshow(sklearn_k1, cmap='gray')
ax[2, 1].imshow(sklearn_k2, cmap='gray')
ax[2, 2].imshow(sklearn_k3, cmap='gray')



plt.savefig("./ImageDeconvolution/Images/lena-periodic.png", dpi=500)
plt.show()
