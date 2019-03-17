import numpy as np
import random
from minfunc import *
from matplotlib import pyplot as plt
data = np.loadtxt('housing.data')
data = data.T
a=np.ones((1,data.shape[1]))

data = np.concatenate((a,data),axis=0)

m =data.shape[1]
n =data.shape[0]
slice = random.sample(list(range(m)),m)
data = data[:,slice]
trainX = data[0:-1,0:400]
trainy = data[-1:,0:400]
testX=data[0:-1,400:]
testy=data[-1:,400:]
theta = np.random.rand(14).reshape(14,1)
#theta = np.ones(14).reshape(14,1)
print(theta.T)
ans = minfunc(theta,trainX,trainy)

pred_y = np.matmul(ans[0].T,testX)
slice=np.argsort(testy)
plt.plot(testy[0,slice].T,'r.')
plt.plot(pred_y[0,slice].T,'b.')
plt.show()



