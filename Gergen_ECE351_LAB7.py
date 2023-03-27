################################################################
#                                                              #
# Sara Gergen                                                  #
# 351-53                                                       #
# Lab 7                                                        #
# Due: 3/7/2023                                                #
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


step = 1e-6

t = np.arange(0, 2*step, step)
#G(s) = (s+9)/(s^2 −6s −16)(s + 4)
#G(s) = (s+9)


#Check the results from task 1
#sig.tf2zpk()


#zeros (s=6)(s-)

steps = 1e-2 # Define step size
t = np . arange (0 , 20 + steps , steps ) 

### Part 1 ### 
#step 1 of part 1#
# G(s) = (s+9)/(s^2 - 6s - 16)(s + 4) = (s+9)/((s-8)*(s+2)*(s+4))
# A(s)= (s+4)/((s+3)*(s+1))
# B(s) = (s+14)*(s+12)


#step 2 of part 1#
zg,pg,gg = sig.tf2zpk([1,9], sig.convolve([1,-6,-16],[1,4]))
print("Poles and Zeroes of G(s)")
print("Zeroes = ",zg)
print("Poles = ",pg)

za, pa, ga = sig.tf2zpk([1,4], [1,4,3])

print("Poles and Zeroes of A(s)")
print("Zeroes = ",za)
print("Poles = ",pa)


broots = np.roots([1,26,168])
print("B(s) Roots")
print("Roots = ",broots)



# step 3 of part 1#
#(s + 9)/(s^2+4s+3)(s^2 - 6s - 16)


#step 5 of part 1#
#numerator = sig.convolve ([1 , 9])
#print ("Numerator = ", numerator )

denom = sig.convolve([1,4,3],[1,-6,-16])
print("Denomanator = ",denom)

openLoop = [1,9],sig.convolve([1,4,3],[1,-6,-16])
t, s = sig.step(openLoop)


plt.figure(figsize=(10,10))
plt.subplot(2,1,1) 
plt.plot(t, s)
plt.grid()
plt.xlabel('t')
plt.title('Open Loop Plot of Transfer function')  


### Part 2 ###


#(numA*denG)/((denG+numB*numG)*denA)

numG = [1,9]
denG = sig.convolve([1,-6,-16],[1,4])

numA = [1,4]
denA = [1,4,3]

numB = [1,26,168]
denB = [1]


num = sig.convolve(numG, numA)
den = sig.convolve(denA, denG) + sig.convolve(sig.convolve(denA, numB), numG)

print("Numerator = ",num)
print("Denom = ",den)

closedLoop = num, den
t, s = sig.step(closedLoop)


plt.figure(figsize = (10,10))
plt.subplot(2,1,1) 
plt.plot(t, s)
plt.grid()
plt.xlabel('t')
plt.title('Closed Loop of Transfer function')  

