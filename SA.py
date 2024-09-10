# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:09:53 2024

@author: yhb
"""

"""
Synchronous averaging
"""

import numpy as np

from csaps import csaps
from scipy import stats

def Corr(a, b):
    n = len(a)
    difval = len(b)-n
    corrarray = np.zeros(difval+1)
    for i in range(difval+1):
        bpart = b[i: i+n]
        corr, _ = stats.pearsonr(a, bpart)
        corrarray[i] = corr
    return corrarray

def func(data, refelist, teen, mul=2):
    """
    Parameters
    ----------
    data : numpy, vibration signal.
    refelist : list, reference point list.
    teen : int, teeth number.
    mul : int, data increase multiple.

    Returns
    -------
    datalist : list, data list after splitting and interpolation of vibration signal.
    atsares : numpy, synchronous averaging signal.
    """
    n_a = refelist[1]
    a = data[:n_a]
    t_a = np.linspace(0, 1, n_a)
    func = csaps(t_a, a)
    t_a_new = np.linspace(0, 1, n_a*mul)
    a_new = func(t_a_new)
    datalist = [a_new]
    #----------
    n = min(len(refelist)-2, 50)  # not too much
    #----------
    for i in range(n):
        sta_b = refelist[i+1]
        sto_b = refelist[i+2]
        n_b_raw = sto_b-sta_b
        sup = int(n_b_raw/teen*0.5)  # parameter
        b = data[sta_b-sup: sto_b+sup]
        n_b = sto_b-sta_b+sup*2
        t_b = np.linspace(0, 1, n_b)
        func = csaps(t_b, b)
        ratio = n_a/n_b_raw
        t_b_new = np.linspace(0, 1, round(n_b*mul*ratio))
        b_new = func(t_b_new)
        corrarray = Corr(a_new, b_new)
        ind = np.argmax(corrarray)
        b_new_cut = b_new[ind:ind+n_a*mul]
        datalist.append(b_new_cut)
    atsares = np.array(datalist).mean(axis=0)
    return datalist, atsares


