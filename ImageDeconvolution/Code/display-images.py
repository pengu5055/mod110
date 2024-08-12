"""
Simply display the images provided in the Supplied Data folder.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import cmasher as cmr
from skimage import io

# Load the images
img1 = io.imread('./SuppliedData/lena_slike/lena_k1_nx.pgm')
img2 = io.imread('./SuppliedData/lena_slike/lena_k2_nx.pgm')
img3 = io.imread('./SuppliedData/lena_slike/lena_k3_nx.pgm')

kernel1 = io.imread('./SuppliedData/lena_slike/kernel1.pgm')
kernel2 = io.imread('./SuppliedData/lena_slike/kernel2.pgm')
kernel3 = io.imread('./SuppliedData/lena_slike/kernel3.pgm')

# Display the images
fig, axs = plt.subplots(2, 3, figsize=(12, 8), layout="compressed")
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('lena_k1_nx.pgm')

norm = mpl.colors.Normalize(vmin=kernel1.min(), vmax=kernel1.max())
axs[1, 0].imshow(kernel1, cmap='gray', norm=norm)
axs[1, 0].set_title('kernel1.pgm')

axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title('lena_k2_nx.pgm')

norm = mpl.colors.Normalize(vmin=kernel2.min(), vmax=kernel2.max())
axs[1, 1].imshow(kernel2, cmap='gray', norm=norm)
axs[1, 1].set_title('kernel2.pgm')

axs[0, 2].imshow(img3, cmap='gray')
axs[0, 2].set_title('lena_k3_nx.pgm')

norm = mpl.colors.Normalize(vmin=kernel3.min(), vmax=kernel3.max())
axs[1, 2].imshow(kernel3, cmap='gray', norm=norm)
axs[1, 2].set_title('kernel3.pgm')

plt.savefig("./ImageDeconvolution/Images/lena_and_kernels.png", dpi=500)
plt.show()
