import numpy as np
import struct
from minfunc2 import *
def loadMNISTImages(filename,picnum):
    f = open(filename, "rb")

    data = f.read(16)
    data = f.read()
    fmt = 'B'
    fmt = fmt*picnum*28*28

    pic = struct.unpack(fmt,data)
    pic = np.asarray(pic)
    pic = pic.reshape(picnum,28,28)
    pic = np.swapaxes(pic,1,2)
    pic = pic.reshape(picnum,28*28).T
    f.close()
    pic = pic/255.0
    return pic


def loadMNISTLabels(filename,picnum):
    f=open(filename,'rb')
    f.read(8)
    data = f.read()
    fmt = 'B'*picnum
    pic = struct.unpack(fmt,data)
    pic = np.asarray(pic).reshape(picnum,1)
    f.close()
    return pic.T

X = loadMNISTImages('G:/PyProjekt/train-images-idx3-ubyte',60000)
y = loadMNISTLabels('G:/PyProjekt/train-labels-idx1-ubyte',60000)

slice = [i for i in range(60000) if y[0,i] == 0 or y[0,i] == 1 ]
X = X[:,slice]
y = y[:,slice]
m = np.mean(X,axis=1).reshape(784,1)
s = np.std(X,axis=1,ddof=1).reshape(784,1)
X = X-m
X = X/(s+0.1)
a=np.ones((1,X.shape[1]))
X = np.concatenate((a,X),axis=0)
train_X = X
train_y = y
X = loadMNISTImages('G:/PyProjekt/t10k-images-idx3-ubyte',10000)
y = loadMNISTLabels('G:/PyProjekt/t10k-labels-idx1-ubyte',10000)
slice = [i for i in range(10000) if y[0,i] == 0 or y[0,i] == 1 ]
X = X[:,slice]
y = y[:,slice]
X = X-m
X = X/(s+0.1)
a=np.ones((1,X.shape[1]))
X = np.concatenate((a,X),axis=0)
test_X = X
test_y = y
theta = np.random.rand(785,1)*5
theta = minfunc(theta,train_X,train_y)
cnt = 0
for i in range(train_y.shape[1]):
    if predict_y(theta,train_X[:,i:i+1]) == int(train_y[0,i]):
        cnt = cnt+1;
accuracy = cnt / float(train_y.shape[1])
print(accuracy)
cnt = 0
for i in range(test_y.shape[1]):
    if predict_y(theta,test_X[:,i:i+1]) == int(test_y[0,i]):
        cnt = cnt+1;
accuracy = cnt / float(test_y.shape[1])
print(accuracy)
