# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:08:25 2024

@author: yhb
"""

"""
Band-pass filtering
"""

import numpy as np

from scipy import signal

def func(L, sf, freqs, amps, phases, speed, teen, coef):
    """
    Parameters
    ----------
    L : int, length of the vibration signal.
    sf : float, sampling frequency.
    freqs : numpy, frequency.
    amps : numpy, amplitude.
    phases : numpy, phase.
    speed : float, speed.
    teen : int, teeth number.
    coef : float, bandwidth multiplier of the band-pass filter.

    Returns
    -------
    harm : int, harmonic order.
    mesvib : numpy, filtered signal.
    ratio : float, peak ratio.
    """
    spefre = speed/60
    mesfrearray = np.zeros(5)
    energyarray = np.zeros(5)
    for i in range(5):
        mesfre = spefre*teen*(i+1)
        mesfrearray[i] = mesfre
        indf = np.argmin(abs(freqs-mesfre+spefre/2))
        indb = np.argmin(abs(freqs-mesfre-spefre/2))
        energyarray[i] = max(amps[indf: indb+1])
    indarray = energyarray.argsort()[::-1]
    #---------
    maxenergy = max(energyarray)
    harmlist = [indarray[0]+1]
    mesfrelist = [mesfrearray[indarray[0]]]
    for i in range(1, 3):
        ind = indarray[i]
        energy = energyarray[ind]
        if energy > maxenergy*0.5:
            harmlist.append(ind+1)
            mesfrelist.append(mesfrearray[ind])

    t =  np.arange(L)/sf
    mesviblist = []
    ratiolist = []
    bandwidth = spefre*coef
    for mesfre in mesfrelist:
        indf = np.argmin(abs(freqs-mesfre+bandwidth/2))
        indb = np.argmin(abs(freqs-mesfre-bandwidth/2))
        mesvib = np.zeros(L)
        for j in range(indf, indb+1):
            freq = freqs[j]
            amp = amps[j]
            phase = phases[j]
            mesvib = mesvib+amp*np.cos(2*np.pi*freq*t+phase)
        #----------
        peaind = signal.argrelextrema(mesvib, np.greater)[0]
        peaval = mesvib[peaind]
        m = peaval.mean()
        s = peaval.std()
        ratio = (m-s*1.645) / (m+s*1.645)  # 90%
        mesviblist.append(mesvib)
        ratiolist.append(ratio)
    ind = np.argmax(ratiolist)
    harm = harmlist[ind]
    mesvib = mesviblist[ind]
    ratio = ratiolist[ind]
    return harm, mesvib, ratio


