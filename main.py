# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:04:35 2024

@author: yhb
"""

import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False


import FFT, ARSE, BPF, RPI, SA


"""health data"""
df = pd.read_csv("health_20160306171241_25600.csv", header=None)
sf = 25600
"""fault data"""
df = pd.read_csv("fault_20170926180904_51200.csv", header=None)
sf = 51200

data = df.values.flatten()


freqs, amps, phases = FFT.func(data, sf)


fc = sum(freqs*amps) / sum(amps)
kf = sum((freqs-fc)**4*amps) / (sum((freqs-fc)**2*amps))**2


"""
Step 1: Average rotational speed estimation.
"""
if kf > 0.1:
    print("The rotational speed is low.")
else:
    t1, t2, t3 = 22, 121, 21
    start, stop = 180, 800
    f, searchrange, spectralenergy, _ = ARSE.func(freqs, amps, t1, t2, t3, start, stop)
    speed = f/t1*60
    print("Average rotational speed is: ", speed)
    
    fig = plt.figure(figsize=(10, 3.5))
    ax = fig.add_subplot(111)
    plt.scatter(searchrange, spectralenergy, s=5)
    plt.axvline(f, color='red', linestyle='--', lw=1, label="Estimated mesh frequency")
    plt.xlabel("Mesh frequency", fontsize=20, fontname="Times New Roman")
    plt.ylabel("Spectral energy", fontsize=20, fontname="Times New Roman")
    plt.xticks(fontsize=16, fontname="Times New Roman")
    plt.yticks(fontsize=16, fontname="Times New Roman")
    plt.legend(prop={'family':'Times New Roman','size':16})
    fig.tight_layout()


"""
Step 2: Band-pass filtering.
"""
L = len(data)
coef = 3
harm, mesvib, ratio = BPF.func(L, sf, freqs, amps, phases, speed, t1, coef)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(211)
plt.plot(data, label="Raw vibration signal")
plt.ylabel("Amplitude", fontsize=20, fontname="Times New Roman")
plt.xticks(fontsize=16, fontname="Times New Roman")
plt.yticks(fontsize=16, fontname="Times New Roman")
plt.legend(prop={'family':'Times New Roman','size':16})
ax = fig.add_subplot(212)
plt.plot(mesvib, label="Filtered vibration signal")
plt.ylabel("Amplitude", fontsize=20, fontname="Times New Roman")
plt.xticks(fontsize=16, fontname="Times New Roman")
plt.yticks(fontsize=16, fontname="Times New Roman")
plt.legend(prop={'family':'Times New Roman','size':16})
fig.tight_layout()



"""
Step 3: Reference points identification.
"""
refelist, speedfluctuation, difference = RPI.func(mesvib, speed, sf, t1, harm)
print("Difference is: ", difference)

fig = plt.figure(figsize=(10, 3.5))
ax = fig.add_subplot(111)
plt.plot(speedfluctuation)
plt.xlabel("Revolution", fontsize=20, fontname="Times New Roman")
plt.ylabel("Speed", fontsize=20, fontname="Times New Roman")
plt.xticks(fontsize=16, fontname="Times New Roman")
plt.yticks(fontsize=16, fontname="Times New Roman")
fig.tight_layout()


"""
Step 3: Synchronous averaging.
"""
_, atsares = SA.func(data, refelist, t1)
fig = plt.figure(figsize=(10, 3.5))
ax = fig.add_subplot(111)
plt.plot(atsares)
plt.ylabel("Amplitude", fontsize=20, fontname="Times New Roman")
plt.xticks(fontsize=16, fontname="Times New Roman")
plt.yticks(fontsize=16, fontname="Times New Roman")
fig.tight_layout()





