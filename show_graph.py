import math
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 12, 100)
z = np.zeros(50)

C = np.cos(t)
S = np.sin(t)
A = np.cos(t[50:])+1
B = np.array([z, A]).reshape(100,)


print("S: {}".format(S.shape))
print("z: {}".format(z.shape))
print("A: {}".format(A.shape))
print("B: {}".format(B.shape))



plt.plot(t[0:50],C[0:50])
plt.plot(t,S)
plt.plot(t,B)
plt.show()
