#two lines to have plots appear in QT 
import matplotlib
matplotlib.use('Qt4Agg')

 
import math
import numpy as np
import matplotlib.pyplot as plt

#let's say times goes from 0 to 12 with 100 discrete steps
t = np.linspace(0, 12, 100)
z = np.zeros(50)#if I want only the beginning

#some curves
C = np.cos(t)
S = np.sin(t)
A = np.cos(t[50:])+1			#nothing at the beginning
B = np.array([z, A]).reshape(100,)	#then a curve

#check sizes
print("S: {}".format(S.shape))
print("z: {}".format(z.shape))
print("A: {}".format(A.shape))
print("B: {}".format(B.shape))

#plot curves
plt.plot(t[0:50],C[0:50])
plt.plot(t,S)
plt.plot(t,B)
plt.show()
