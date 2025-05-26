import numpy as np
import scipy.ndimage

def window_std(I):
    window = np.ones(shape=(5, 5))/25
    K_I2 = scipy.ndimage.convolve(I**2, window, mode='reflect')
    KI_2 = scipy.ndimage.convolve(I, window, mode='reflect')**2
    return np.sqrt(K_I2 - KI_2)