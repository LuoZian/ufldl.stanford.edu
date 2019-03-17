import numpy as np
import struct
from minfunc3 import *
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
X = X-m
X = X/(s+0.1)
a=np.ones((1,X.shape[1]))
X = np.concatenate((a,X),axis=0)
test_X = X
test_y = y
theta = np.random.rand(785,10)*0.5
theta = minfunc(theta,train_X,train_y)
print( predict_y(theta,train_X,train_y))
print( predict_y(theta,test_X,test_y))
