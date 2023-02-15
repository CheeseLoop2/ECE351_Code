################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 4                                                        #
# Due: 2/14/2023                                               #
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
low = -10
up = 10+step 
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

t = np.arange(-10,10,step)

def h1(t):
    for i in range(len(t)): 
      #  y = (u(t)-u(t-3))*(math.e**(-2t))
         y = np.exp(-2*t)*(u(t)-u(t-3)) #implied multiplication trouble ^
    return y
    y = np.zeros(t.shape)
    

def h2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y = u(t-2) - u(t-6)
    return y

def h3(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        f = 0.25
        w = 2*np.pi*f
        y = np.cos(w*t)*u(t)
    return y

y = h1(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Function 1')

y = h2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Function 2')


y = h3(t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Function 3')


#Part 2 Begin 



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

x = np.arange(-20, 2*t[Nt-1]+step, step)
#x = np.append(t,np.zeros((1,Nt-1)))

y = conv(h1(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y*step)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 1 with Step')

y = conv(h2(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y*step)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 2 with Step')

y = conv(h3(t),u(t))

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(x, y*step)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Convolved Function 3 with Step')


#Checking hand derived equations
    


def dh1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)): 
        
        y = 0.5*(1 - np.exp(-2*t))*(u(t)) - 0.5*(1 - np.exp(-2*(t-3)))*u(t-3)
     
    return y
 
t = np.arange(-10,10,step)
y = dh1(t)


plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('hand Derived Function of h1')




def dh2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)): 
        
        y = (t-2)*u(t-2) - (t-6)*u(t-6)
     
    return y
 
#t = np.arange(-20,20,step)
y = dh2(t)


plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('hand Derived Function of h2')



def dh3(t):
    y = np.zeros(t.shape)
    f = 0.25
    w = 2*np.pi*f
    
    for i in range(len(t)): 
        
         y = (1/w)*(np.cos(w*t)*np.sin(w*t) - np.sin(w*t)*(np.cos(t*w)-1))*u(t)
        
    return y
 
t = np.arange(-10,10,step)
y = dh3(t)


plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('hand Derived Function of h3')
