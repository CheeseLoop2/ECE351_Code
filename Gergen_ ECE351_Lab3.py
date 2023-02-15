################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 3                                                        #
# Due: 2/7/2023                                                #
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


step = 1e-1
low = 0
up = 20+step 
dif = up-low
t = np.arange(low,up,step)

#Defining step function 

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

#Ramp Function definition

def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
       
        if t[i] >= 0:
            y[i] = t[i] 
        else:
            y[i] = 0
    return y

#Part 1;

def f1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y = u(t-2) - u(t-9)
    return y


def f2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)): 
        y = math.e**(-t)*u(t) #must import math in order to use this 
    return y
        
def f3(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y = r(t-2)*(u(t-2)-u(t-3)) + r(4-t)*(u(t-3)-u(t-4))
    return y


y = f1(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Function 1')


t = np.arange(-5, 10+step, step)
y = f2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Function 2')


t = np.arange(low,up,step)

y = f3(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Function 3')

#Part 2; 



def conv(f1,f2):
    
    NF1 = len(f1)
    NF2 = len(f2)
        
    f1Extended = np.append(f1,np.zeros((1,NF2-1)))
    
    f2Extended = np.append(f2,np.zeros((1,NF1-1)))
    
    result = np.zeros(f1Extended.shape)
    
    
    for i in range(NF2 + NF1 - 1):
        
        result[i] = 0
    
        for j in range(NF1):
            #check for errors
            if ( (i - j) + 1 > 0):
        
                try:
                    result[i] += f1Extended[j] * f2Extended[i-j+1] 
                    
                except:
                    print(i-j)
            
    return result
    

Nt = len(t)

x = np.append(t,np.zeros((1,Nt-1)))

y = conv(f1(t),f2(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 1 and Function 2 ')


y = conv(f2(t),f3(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 2 and Function 3 ')


#F1 and F3
y = conv(f1(t),f3(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 1 and Function 3')






 
