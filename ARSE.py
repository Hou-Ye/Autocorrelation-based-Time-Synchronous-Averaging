# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:05:24 2024

@author: yhb
"""

"""
Averaging rotational speed estimation
"""

import numpy as np

def func(freqs, amps, t1, t2, t3, start, stop, n=3, w=1.6):
    """
    Parameters
    ----------
    freqs : numpy, frequency.
    amps : numpy, amplitude.
    t1 : int, high-speed shaft gear teeth number, 22.
    t2 : int, intermediate-speed shaft large gear teeth number, 112.
    t3 : int, intermediate-speed shaft small gear teeth number, 21.
    start : float, mesh frequency search range start point, start~speed_min/60*teeth_number.
    stop : float, mesh frequency search range stop point, stop~speed_max/60*teeth_number.
    step : float, mesh frequency search step.
    n : int, harmonic order.
    w : float, weight.
    
    Returns
    -------
    f : float, estimated mesh frequency.
    searchrange : numpy, mesh frequency search range.
    spectralenergy : numpy, spectral energy.
    auxiliaryresult : numpy, auxiliary result.
    """
    ind = np.argmin(abs(freqs-stop*n))
    freqs = freqs[:ind+5]
    amps = amps[:ind+5]
    
    dpi = freqs[1]
    step = dpi / n
    
    searchrange = np.arange(start, stop, step)
    spectralenergy = []
    auxiliaryresult = []
    for i in searchrange:
        spefreind = np.argmin(abs(freqs-i/t1))
        energy = amps[spefreind]
        
        temp = [energy]
        
        inde_1 = np.argmin(abs(freqs-i))
        energy = amps[inde_1]**w*energy
        inde_2 = np.argmin(abs(freqs-i/t2*t1))
        energy = amps[inde_2]**w*energy
        
        temp = temp+[amps[inde_2]**w, amps[inde_1]**w]
        
        for j in range(1,n):
            inde_1 = np.argmin(abs(freqs-i*(j+1)))
            energy = energy*amps[inde_1]
            inde_2 = np.argmin(abs(freqs-i/t2*t1*(j+1)))
            energy = energy*amps[inde_2]
            
            temp = temp+[amps[inde_2], amps[inde_1]]
            
        spectralenergy.append(energy)
        auxiliaryresult.append(temp)
        
    f = searchrange[np.argmax(spectralenergy)]
    
    auxiliaryresult = np.array(auxiliaryresult)
    return f, searchrange, spectralenergy, auxiliaryresult






