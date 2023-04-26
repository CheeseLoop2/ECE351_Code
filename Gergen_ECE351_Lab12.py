################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 12 : Final Lab                                           #
# Due: 4/25/2023                                               #
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
from matplotlib import patches


#graph the given signal:
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10, 7 ))
plt.plot(t, sensor_sig)
plt.title("Noisy Input Signal")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.grid()
plt.show()


##############Establishing Functions##############
## Ploting ##
def make_stem(ax ,x ,y , color ='k', style ='solid' , label ='', linewidths =2.5 ,** kwargs ) :
    ax.axhline( x[0] , x[ -1] ,0 , color = 'r')
    ax.vlines(x , 0 ,y , color = color , linestyles = style , label = label , linewidths =
    linewidths)
    ax.set_ylim([1.05* y.min(), 1.05*y.max() ])


fs = 1e6

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


############## END FUNCTIONS ##############


freq, xmag, xphi = fastfunction2(sensor_sig, fs)

#Entire range
fig, ax = plt.subplots(figsize = (10, 3))
plt.title('Entire Range')
plt.xlim(0,5e5)
make_stem(ax, freq, xmag)
plt.ylabel('Magnitude')
plt.xlabel('frequency [Hz]')
plt.show()

fig, ax = plt.subplots(figsize = (10, 3))
plt.title('Entire Range')
plt.xlim(0,5e5)
make_stem(ax, freq, xphi)
plt.xlabel('frequency [Hz]')
plt.ylabel('Phase')
plt.show()


##############Filtering##############

##Using code from lab 10##

#Using Values found

R = 750e3
L = 31.9
C = 220e-12 

numerator = [0, R/L, 0]
denominator = [1, R/L, 1/(L*C)]

# find w for stuff
# step size, start, and stop



# transfer function
Transfer = con.TransferFunction(numerator, denominator)

#Entire bode plot
w = np.arange(0, 5e7, 10)
plt.figure(figsize = (10, 7))
plt.title("Entire Bode")
_ = con.bode(Transfer, w, dB=True, Hz=True, deg=True, plot=True)

#Low Freq bode plot
w2 = np.arange(1, 1.8e3, 10)*2*np.pi
plt.figure(figsize = (10, 7))
plt.title("Low Freqencies")
_ = con.bode(Transfer, w2, dB=True, Hz=True, deg=True, plot=True)

#Position frequencies
w3 = np.arange(1.8e3, 2e3, 10)*2*np.pi
plt.figure(figsize = (10, 7))
plt.title("Position Frequencies")
_ = con.bode(Transfer, w3, dB=True, Hz=True, deg=True, plot=True)

# high frequencies
w4 = np.arange(2e3, 1e6, 10)*2*np.pi
plt.figure(figsize = (10, 7))
plt.title("High Freqencies")
_ = con.bode(Transfer,w4, dB=True, Hz=True, deg=True, plot=True)


####### Filtering the signal #######

#z-transforming the signal
zNum, pDen = sig.bilinear(numerator,denominator,fs)

#After z-tranforming it, put it through the lfilter function
signalFiltered = sig.lfilter(zNum, pDen, sensor_sig)

#Plotting the filtered signal
plt.figure(figsize = (10,7))
plt.title("Filtered Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.plot(t,signalFiltered)
plt.grid()
plt.show()



######## Fast Function and plotting the Filtered Frequencies #########

Filtered_Frequency, Filtered_Magnitude, Filtered_phase = fastfunction2(signalFiltered, fs)

fig2 = plt.figure(figsize = (10,10), constrained_layout = True)

fig, ax = plt.subplots(figsize = (10, 3))
plt.title('Entire Range')
plt.xlim(0,5e5)
make_stem(ax, Filtered_Frequency, Filtered_Magnitude)
plt.ylabel('Magnitude')
plt.xlabel('frequency [Hz]')
plt.show()

fig, ax = plt.subplots(figsize = (10, 3))
plt.title('Low Range')
plt.xlim(0,1.79e3)
make_stem(ax, Filtered_Frequency, Filtered_Magnitude)
plt.ylabel('Magnitude')
plt.xlabel('frequency [Hz]')
plt.show()


fig, ax = plt.subplots(figsize = (10, 3))
plt.title('Position Range')
plt.xlim(1.8e3, 2e3)
make_stem(ax, Filtered_Frequency, Filtered_Magnitude)
plt.ylabel('Magnitude')
plt.xlabel('frequency [Hz]')
plt.show()


fig, ax = plt.subplots(figsize = (10, 3))
plt.title('High Frequency Range')
plt.xlim(2.1e3,5e5)
make_stem(ax, Filtered_Frequency, Filtered_Magnitude)
plt.ylabel('Magnitude')
plt.xlabel('frequency [Hz]')
plt.show()
