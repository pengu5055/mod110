"""
Homemade and Stolen Deconvolution Algorithms
"""
import numpy as np
import scipy.fft as fft

def H_deconvolve(star, psf):
    star_fft = fft.fftshift(fft.fftn(star))
    psf_fft = fft.fftshift(fft.fftn(psf))
    return fft.fftshift(fft.ifftn(fft.ifftshift(star_fft/psf_fft)))