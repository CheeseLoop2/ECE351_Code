################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 6                                                        #
# Due: 2/28/2023                                               #
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


step = 1e-2

t = np.arange(0, 2+step, step)


#Define Step Function
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y


### Part 1 ###

#step 1

y = (np.exp(-6*t)+2*np.exp(-4*t))*u(t)


plt.figure(figsize = (12, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Step Response Hand Calculated')



def stepFuncFreq(s):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        y = 1/(s+6)- 0.5/(s+4)+0.5/s
    
    return y


#step 2 of part 1#


#step response
numerator = [1,6,12]
denom = [1,10,24]


t, ys = sig.step((numerator,denom),T=t)

plt.figure(figsize = (12, 7))
plt.subplot(2,1,2) 
plt.plot(t,ys)
plt.grid()
plt.ylabel('Response with scipy') 
plt.xlabel('t')
plt.title('Scipty Signal Response')


#Step 3 of part 1

R,P,K = sig.residue(numerator,denom)

print("R = ", R)
print("P = ", P)
print("K = ", K)



tmax = 4.5
deny2 = [1,18,218,2036,9085,2550]
numx2 = [25250] 

R2, P2, K2 = sig.residue(numx2,deny2)
print("R=",R2)
print("P=",P2)
print("K=",K2)

def cosMethod(R,P,t):
    y = 0
    for i in range(len(R)):
        y += (abs(R[i])*np.exp(np.real(P[i])*t)*np.cos(np.imag(P[i]*t)+np.angle
        (R[i]))*u(t))
    return y

y_1 = cosMethod(R2,P2,t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,y_1)
plt.grid()
plt.xlabel('t')
plt.title('Step Response of given System; Cosine Method')  



xCof = [25250]
yCof = [1,18,218,2036,9085,25250]

t,ys = sig.step((xCof,yCof),T=t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,ys)
plt.grid()
plt.xlabel('t')
plt.title('Step Response using sig.step()')  

