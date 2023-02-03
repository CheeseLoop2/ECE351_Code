# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:50:59 2023

@author: Sara
"""

################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 2                                                        #
# Due: 1/31/2023                                               #
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


# Set font size in plots
#plt.rcParams.update({'fontsize': 14}) 


# Define step size:
steps = 1e-2

#graph bounds

UpperBound = 10 + steps
LowerBound = -5

#Getting the hashmarks
Difference = UpperBound - LowerBound


#Add step size to make sure the plot includes 5.0
# since np.rang() only goes up to second argument
t = np.arange(LowerBound, UpperBound + steps, steps)


def funCos(t): # The only variable sent to the function is t
    fun1 = np.zeros(t.shape) # initialze y(t) as an array of zeros

    for i in range(len(t)): # run the loop once for each index of t
               
            fun1[i] = np.cos(t[i]) 
   
    return fun1 #send back the output stored in an array

fun1 = funCos(t) # call the function we just created

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, fun1)
plt.grid()
plt.xlabel('t')
plt.ylabel("FunCos(t)")
plt.title("Cosine Graph;Task 2’")


###########Part 2 of the lab####################################3


#Step function:
    
    
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

#demonstrating the following functions
t = np.arange(-5, UpperBound + steps, steps)
def step(t):
    
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        if t[i] > 0:
            y = u(t)
            
        else:
            
            y = 0
    return y

y = step(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t,y)
plt.grid()
plt.xlabel('t')
plt.ylabel("Y(t)")
plt.title("Step Function")


#ramp function
t = np.arange(-5, UpperBound + steps, steps)
#demonstrating the following functions
def ramp(t):
    
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        
        if t[i] > 0:
            y = r(t)
            
        else:
            
            y = 0
    return y

y = ramp(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t,y)
plt.grid()
plt.xlabel('t')
plt.ylabel("Y(t)")
plt.title("Ramp Function")


t = np.arange(LowerBound, UpperBound + steps, steps)

def fun2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):        
        
        y = r(t) - r(t-3)+ 5*u(t-3) - 2*u(t-6) - 2*r(t-6)
        
        return y


y = fun2(t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t,y)
plt.grid()
plt.xlabel('t')
plt.ylabel("Y(t)")
plt.title("Plot for Lab 2")


y = -fun2(t)

plt.figure(figsize = (10, 7))
#plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Reversal Plot for Part 3')

# #Task 2 time shifts f(t-4) and f(-t-4)

t = np.arange(0, 14 + steps, steps)

y = fun2(t-4)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('f(t-4) Plot')

t = np.arange(-15, 0 + steps, steps)
y = fun2(-t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('f(-t-4) Plot')

# #Task 3 time scale operations f(t/2) and f(2t)
t = np.arange(0, 20 + steps, steps)
y = fun2(t/2)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('f(t/2)')

t = np.arange(0, 5 + steps, steps)
y = fun2(2*t)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(2t)')


#derivative numpy.diff() 

#steps = 1e-3
t = np.arange(-5, 10 + steps, steps)
arr = np.array(fun2(t))
dt = np.diff(arr)
dy = np.diff(arr)

plt.figure(figsize = (10, 7))
plt.plot(t[:-1], dy/dt)
plt.ylim(-3,10)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Derivative Plot')





