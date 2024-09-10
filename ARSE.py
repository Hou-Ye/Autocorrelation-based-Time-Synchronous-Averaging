# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:05:24 2024

@author: yhb
"""

"""
Averaging rotational speed estimation
"""

import numpy as np

def func(freqs, amps, t1, t2, t3, start, stop, step=0.3667, n=3, offset_1=2, offset_2=3, w=1.4):
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
    offset_1 : int, number of spectrum lines added on both sides of the mesh harmonic frequencie, l1.
    offset_2 : int, number of spectrum lines added on both sides of the mesh harmonic frequencie, l2.
    w : float, weight.
    
    Returns
    -------
    f : float, estimated mesh frequency.
    searchrange : numpy, mesh frequency search range.
    spectralenergy : numpy, spectral energy.
    auxiliaryresult : numpy, auxiliary result.
    """
    ind = np.argmin(abs(freqs-stop*n))
    freqs = freqs[:ind+offset_1+5]
    amps = amps[:ind+offset_1+5]
    
    searchrange = np.arange(start, stop, step)
    spectralenergy = []
    auxiliaryresult = []
    for i in searchrange:
        spefreind = np.argmin(abs(freqs-i/t1))
        energy = amps[spefreind]
        
        temp = [energy]
        
        inde_1 = np.argmin(abs(freqs-i))
        energy = (sum(amps[inde_1-offset_1: inde_1+offset_1+1]))**w*energy
        inde_2 = np.argmin(abs(freqs-i/t2*t1))
        energy = (sum(amps[inde_2-offset_2: inde_2+offset_2+1]))**w*energy
        
        temp = temp+[(sum(amps[inde_2-offset_2: inde_2+offset_2+1]))**w, (sum(amps[inde_1-offset_1: inde_1+offset_1+1]))**w]
        
        for j in range(1,n):
            inde_1 = np.argmin(abs(freqs-i*(j+1)))
            energy = energy*sum(amps[inde_1-offset_1: inde_1+offset_1+1])
            inde_2 = np.argmin(abs(freqs-i/t2*t1*(j+1)))
            energy = energy*sum(amps[inde_2-offset_2: inde_2+offset_2+1])
            
            temp = temp+[sum(amps[inde_2-offset_2: inde_2+offset_2+1]), sum(amps[inde_1-offset_1: inde_1+offset_1+1])]
            
        spectralenergy.append(energy)
        auxiliaryresult.append(temp)
        
    f = searchrange[np.argmax(spectralenergy)]
    
    auxiliaryresult = np.array(auxiliaryresult)
    return f, searchrange, spectralenergy, auxiliaryresult






