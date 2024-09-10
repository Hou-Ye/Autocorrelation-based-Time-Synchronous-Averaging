# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:09:02 2024

@author: yhb
"""

"""
Reference points identification
"""

import numpy as np

from scipy import signal, stats

def Corr(a, b):
    n = len(a)
    difval = len(b)-n
    corrarray = np.zeros(difval+1)
    for i in range(difval+1):
        bpart = b[i: i+n]
        corr, _ = stats.pearsonr(a, bpart)
        corrarray[i] = corr
    return corrarray

def func(mesvib, speed, sf, teen, harm):
    """
    Parameters
    ----------
    mesvib : numpy, filtered signal.
    speed : float, speed.
    sf : float, sampling frequency.
    teen : int, teeth number.
    harm : int, harmonic order.

    Returns
    -------
    refelist : list, reference point list.
    speedfluctuation : numpy, speed fluctuation.
    difference : int, difference
    """
    cyc = int(60/speed*sf)
    inter = int(60/speed/teen*sf)
    refe = 0
    refelist = [refe]
    try:
        a = mesvib[refe: refe+cyc*2]
        b = mesvib[refe: refe+cyc*3+inter*2]
        corrarray = Corr(a, b)
        max_peaks = signal.argrelextrema(corrarray, np.greater)[0]
        cyc = max_peaks[teen*harm-1]
        refe = refe+cyc
        refelist.append(refe)
    except:
        a = mesvib[refe: refe+cyc*2]
        b = mesvib[refe: refe+cyc*3+inter*4]
        corrarray = Corr(a, b)
        max_peaks = signal.argrelextrema(corrarray, np.greater)[0]
        cyc = max_peaks[teen*harm-1]
        refe = refe+cyc
        refelist.append(refe)
    #---------- 2~n
    while len(mesvib[refe: ]) > cyc*3+inter*4:
        try:
            a = mesvib[refe: refe+cyc*2]
            b = mesvib[refe: refe+cyc*3+inter*2]
            corrarray = Corr(a, b)
            max_peaks = signal.argrelextrema(corrarray, np.greater)[0]
            cyc = max_peaks[teen*harm-1]
            refe = refe+cyc
            refelist.append(refe)
        except:
            a = mesvib[refe: refe+cyc*2]
            b = mesvib[refe: refe+cyc*3+inter*4]
            corrarray = Corr(a, b)
            max_peaks = signal.argrelextrema(corrarray, np.greater)[0]
            cyc = max_peaks[teen*harm-1]
            refe = refe+cyc
            refelist.append(refe)
    speedfluctuation  = np.array([sf/(refelist[i+1]-refelist[i])*60 for i in range(len(refelist)-1)])
    difference = max([abs(refelist[i+2]+refelist[i]-refelist[i+1]*2) for i in range(len(refelist)-2)])
    return refelist, speedfluctuation, difference


