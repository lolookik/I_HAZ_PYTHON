#system settings for plots
import matplotlib
matplotlib.use('Qt4Agg')
 
#basic math/plotting imports
import math
import numpy as np
import matplotlib.pyplot as plt

#importing LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

#let us suppose that times goes from 0 to tmax with steps steps
tmax = 10
steps = 10
t = np.linspace(0, tmax, steps)
z = np.zeros(int(tmax/2))#if I want only the beginning

#some curves
C = np.cos(t)
S = np.sin(t)
data = np.vstack([C,S])
data = data.T
A = np.cos(t[int(tmax/2):])+1			#nothing at the beginning
B = np.array([z, A]).reshape(steps,)	#then a curve

#let's do models
model = Sequential()
model.add(LSTM(32, input_shape=(1,100), return_sequences=False))
model.add(Dense(32))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


#check sizes
print("S: {}".format(S.shape))
print("z: {}".format(z.shape))
print("A: {}".format(A.shape))
print("B: {}".format(B.shape))
print(data.shape)
print(data)

#plot curves
plt.plot(t[0:int(tmax/2)],C[0:int(tmax/2)])
plt.plot(t,S)
plt.plot(t,B)
plt.show()
