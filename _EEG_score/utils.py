import torch
import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.decomposition import PCA, FastICA

def my_fft(data):
    # data shape [length]
    yf1 = abs(fft(data))/len(data)
    yf2 = yf1[range(int(len(data)/2))]

    xf = np.arange(0, int(len(data)/2))
    return xf, yf2

def ICA_(data):
    components = 3
    data = np.array(data)
    ica = FastICA(n_components=components)
    result = None
    for lines in data:
        ica_fit = ica.fit(lines)[None, ...]
        result = ica_fit if result is None else result.append(ica_fit, axis=0)
    return result  # []