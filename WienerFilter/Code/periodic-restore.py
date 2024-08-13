"""
Restore the images with periodic perturbations added. *_nx.pgm
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import matplotlib as mpl
from skimage import io
import scipy.signal as signal
from skimage import color, data, restoration

# Load the images
lena_k1 = io.imread('./SuppliedData/lena_slike/lena_k1_nx.pgm')
kernel1 = io.imread('./SuppliedData/lena_slike/kernel1.pgm')
lena_k2 = io.imread('./SuppliedData/lena_slike/lena_k2_nx.pgm')
kernel2 = io.imread('./SuppliedData/lena_slike/kernel2.pgm')
lena_k3 = io.imread('./SuppliedData/lena_slike/lena_k3_nx.pgm')
kernel3 = io.imread('./SuppliedData/lena_slike/kernel3.pgm')

# Create a 2D window
window_1d = signal.windows.gaussian(kernel1.shape[0], std=120)
window_2d = np.outer(window_1d, window_1d)


# Display the images
fig, ax = plt.subplots(3, 3, figsize=(10, 8))
for a in ax.flatten():
    a.axis('off')

ax[0, 0].imshow(lena_k1, cmap='gray')
ax[0, 0].set_title('lena_k1_nx.pgm')
ax[0, 1].imshow(lena_k2, cmap='gray')
ax[0, 1].set_title('lena_k2_nx.pgm')
ax[0, 2].imshow(lena_k3, cmap='gray')
ax[0, 2].set_title('lena_k3_nx.pgm')


plt.savefig("./ImageDeconvolution/Images/lena-periodic.png", dpi=500)
plt.subplots_adjust(wspace=0.8)
plt.tight_layout()
plt.show()
