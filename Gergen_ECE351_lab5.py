################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 5                                                        #
# Due: 2/21/2023                                               #
# Any other necessary information needed to navigate the file  #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import pandas as pd
import control
import time
from scipy.fftpack import fft , fftshift
import math


### Part 1 ###

step = 1e-6

t = np.arange(0, 1.2e-3, step)

#impulse response h(t) hand calculated

ht = 10000*np.e**(-5000*t)*np.cos(18584.133*t) - 2690.467*np.e**(-5000*t)*np.sin(18584.133*t)


plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.title('Hand Calculated')
plt.plot(t, ht)
plt.ylabel('h(t)')
plt.xlabel('t')
plt.grid()



R = 1000 
L = 27E-3
C = 100E-9


z = 1/(R*C)
a = -0.5*z
w = 0.5*np.sqrt(z**2 - 4/(L*C)+0*1j)

#fancy functions:
p = a + w
g = p * z

g = np.sqrt((-0.5*z**2)**2 + (z*w)**2)
g_abs = np.abs(g)
g_ang = np.angle(g)


#transfer function 
#H(s) = 10000s/(s^2+(s/10000)+3.7037*10^8)

system = ([10000.0, 0.0], [1.0, 10000.0, 370370370.4])
tout, yout = sig.impulse(system) #impulse using scipy


plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.title('Scipy.signal.impulse() function')
plt.plot(tout, yout)
plt.ylabel('h(t)')
plt.xlabel('t')
plt.grid()


#### Part 2 ###

tout,yout = sig.step(system)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.title('Step response of H(s) using scipy.step')
plt.plot(tout, yout)
plt.grid()
plt.ylabel('H(t)')
plt.xlabel('t')