# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:35:22 2024

@author: yhb
"""

"""
fast Fourier transform
"""

import numpy as np

from scipy.fftpack import fft, fftfreq

def func(data, sf):
    """
    Parameters
    ----------
    data : numpy, vibration signal.
    sf : float, sampling frequency.
    
    Returns
    -------
    freqs : numpy, frequency.
    amps : numpy, amplitude.
    phases : numpy, phase.
    """
    data = np.array(data).flatten()
    L = len(data)
    T = 1/sf
    comp = fft(data)
    doublefreqs = fftfreq(L, T)
    absamps = np.abs(comp)
    normalamps = absamps / (L/2)
    normalamps[0] /= 2
    phaseangles = np.angle(comp)
    freqs = doublefreqs[doublefreqs >= 0]
    amps = normalamps[doublefreqs >= 0]
    phases = phaseangles[doublefreqs >= 0]
    return freqs, amps, phases


