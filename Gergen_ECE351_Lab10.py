################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 10                                                       #
# Due: 4/4/2023                                                #
#                                                              #
################################################################

import control as con
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import pandas as pd
import control
import time
from scipy.fftpack import fft , fftshift
import math

#steps
step = 1

w = np.arange(1e3, 1e6*step, step)

R = 1000
C = 27*10**-3
L = 100*10**-9

###Part 1, task 1###
#make sure it is in dB

HMagn = np.sqrt((w/R*C)**2)/np.sqrt((-1*w**2+(1/(L*C)))**2+(w/(R*C))**2)

Hangle = ((np.pi/2) - np.arctan( (w/(R*C)) / (-1*(w**2) + (1/(L*C))) ))


db_magn = 20*np.log10(HMagn)
DegAngle =  Hangle*180/np.pi

##Fixing the angle so the phase graph swoops down correctly
for i in range(len(DegAngle)):
    if (DegAngle[i] > 90):
        DegAngle[i] = db_magn[i] - 180


plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
plt.semilogx(w, db_magn)
plt.title("Pre-lab Hand Calculations")
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogx(w, DegAngle)
plt.grid()

##Part 1 task 2##
##use .bode to plot the magnitude and phase

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

sys = sig.TransferFunction ( num , den )
w2, magn2, phase2 = sig.bode(sys, w,)

plt.figure(figsize=(10,7))
plt.title("Python Function")
plt.subplot(2, 1, 1)
plt.semilogx(w2, magn2)
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogx(w2, phase2)
plt.grid()

##Part 1, task 3##
##PLot the freq in Hz


system = con.TransferFunction ( num , den )
_ = con.bode ( system, w, dB = True , Hz = True , deg = True , Plot = True )

plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
plt.semilogx(w2, magn2)
plt.title("Frequency in Hertz")
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogx(w2, phase2)
plt.grid()


##Part 2##

##Part 2, task 1##

fs = 1000000
step = 1/fs
t = np.arange(0, 0.01, step)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
plt.title("Output Signal x(t)")
plt.plot(t, x)
plt.grid()

First, second = sig.bilinear(num, den, fs)

y = sig.lfilter(First, second, x)

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()

