################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 9                                                        #
# Due: 3/28/2023                                               #
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

#steps
step = 1e-2
t = np.arange(0, 20+step, step)
x = np.cos(3*np.pi*t)
fs = 100

 ### Part 1 ###  


def fastfunction(x, fs):
    N = len(x)
    X_fft = sp.fftpack.fft(x)
    X_fft_shifted = sp.fftpack.fftshift(X_fft)
        
    freq = np.arange(-N/2, N/2)*fs/N
        
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
            
    return freq, X_mag, X_phi   
    
freq, X_mag, X_phi = fastfunction(x, fs)


###Task 1###

T = 1/fs
t = np.arange(0 , 2, T)

x = np.cos(2*np.pi*t)


plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Task 1 $x(t) = cos(2 \pi t)$")


freq, X_mag, X_phi = fastfunction(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        
               
#plt . stem ( freq , X_mag ) # you will need to use stem to get these plots to be
#plt . stem ( freq , X_p


##Task 2##

x = 5*np.sin(2*np.pi*t)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Task 2 $x(t) = 5sin(2 \pi t)$")


freq, X_mag, X_phi = fastfunction(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        
           
##Task 3##

x = 2*np.cos((2*np.pi*2*t) - 2) + np.sin((2*np.pi*6*t) + 3)**2

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Cosine Task 3")


freq, X_mag, X_phi = fastfunction(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        


##Task 4##

def fastfunction2(x, fs):
    N = len(x)
    X_fft = sp.fftpack.fft(x)
    X_fft_shifted = sp.fftpack.fftshift(X_fft)
        
    freq = np.arange(-N/2, N/2)*fs/N
        
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
            
    return freq, X_mag, X_phi   
    
freq, X_mag, X_phi = fastfunction(x, fs)

x = np.cos(2*np.pi*t)


plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Task 4(1) $x(t) = cos(2 \pi t)$")


freq, X_mag, X_phi = fastfunction2(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        
               

##Task 2##

x = 5*np.sin(2*np.pi*t)

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Task 4(2) $x(t) = 5sin(2 \pi t)$")


freq, X_mag, X_phi = fastfunction2(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        
           
##Task 3##

x = 2*np.cos((2*np.pi*2*t) - 2) + np.sin((2*np.pi*6*t) + 3)**2

plt.figure(figsize = (10 ,7))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Cosine Task 4(3)")


freq, X_mag, X_phi = fastfunction2(x, fs)

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)        


##Task 5##


#Run for the Fourier series approximation of the square wave plotted in Lab 8, using only the
#N = 15 case through your clean version of the fft developed in Task 4. For the time-domain
#plot, use 0 ≤ t ≤ 16s. Further, use T = 8 for the period of the series as you did in Lab 8.

T1 = 8
w = 2*np.pi/T1
steps = 1e-3
t=np.arange(0, 16, T)

def function(T,t):
    y = 0
    for k in range(15):
        b = 2/((k+1)*np.pi) * (1-np.cos((k+1)*np.pi))
        x = b * np.sin((k+1)*w*t)
        
        y += x
    return y

x = function(T1, t)

freq, X_mag, X_phi = fastfunction2(x, fs)

plt.figure(figsize = (10 ,7))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title("Task 5 - N=15")

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.grid()
plt.xlim(-3, 3)

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi) 
plt.grid()
plt.xlim(-3, 3)
plt.ylabel('Phase')

plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.grid()
plt.xlim(-3, 3)