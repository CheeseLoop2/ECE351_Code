################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 8                                                        #
# Due: 3/21/2023                                                #
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

#steps
step = 1e-2
t = np.arange(0, 20+step, step)

#Define Step Function
def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

#define variables:

    
#Part 1 Task 1#

#when first 
k = 1 #a0
ak = 0
bk = (2/(k*np.pi))*((1-np.cos(k*np.pi)))
print("k = ", k)
print("a1 = ", ak)
print("b1 = ",bk)


#Second term: k = 2
k = 2
ak = 0
bk = (2/(k*np.pi))*((1-np.cos(k*np.pi)))
print("k = ", k)
print("a2 = ", ak)
print("b2 = ",bk)


#Third term: k =3
k = 3
ak = 0
bk = (2/(k*np.pi))*((1-np.cos(k*np.pi)))
print("k = ", k)
print("a3 = ", ak)
print("b3 = ",bk)


#Part 1 task 2#

T = 8 #seconds

w = 2*np.pi/T


#N = {1, 3, 15, 50, 150, 1500}
n0 = 1
n1 = 3
n2 = 15
n3 = 50
n4 = 150
n5 = 1500


for k in range(1,1+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n0 = bk*np.sin(k*w*t)

for k in range(1,3+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n1 += bk*np.sin(k*w*t)
    
for k in range(1,15+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n2 += bk*np.sin(k*w*t)
    
for k in range(1,50+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n3 += bk*np.sin(k*w*t)

for k in range(1,150+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n4 += bk*np.sin(k*w*t)
    
for k in range(1,1500+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n5 += bk*np.sin(k*w*t)

#plotting


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t, n0)
plt.title('N = 1')
plt.grid()


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t, n1)
plt.title('N = 3')
plt.grid()


plt.figure(figsize = (10,10))
plt.subplot(3,1,2)
plt.plot(t,n2)
plt.title('N = 15')
plt.grid()


plt.figure(figsize = (10,10))
plt.subplot(3,1,3)
plt.plot(t,n3)
plt.title('N = 50')
plt.grid()


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
plt.plot(t, n4)
plt.title('N = 150')
plt.grid()


plt.figure(figsize = (10,10))
plt.subplot(3,1,2)
plt.plot(t,n5)
plt.title('N = 1500')
plt.grid()
