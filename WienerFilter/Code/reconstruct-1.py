"""
Reconstruct the signal signal1.dat using the Wiener filter.
The expected output is signal0.dat
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import scipy.fft as fft
import cmasher as cmr