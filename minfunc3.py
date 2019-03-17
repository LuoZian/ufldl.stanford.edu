import numpy as np


def getfg(theta,X,y):
    p = np.exp(X.T.dot(theta))
    p=p/np.sum(p,axis=1).reshape(-1,1)
    sparse = np.zeros(p.shape)
    for i in range(10):
        slice = (y.reshape(-1,1) == i)
        sparse[:,i:i+1] = slice
    g = p - sparse
    g = X.dot(g)
    return g
def minfunc(theta,X,y):
    for i in range(100):
       g = getfg(theta,X,y)
       theta = theta -g*0.00005
    return theta
def predict_y(theta,X,y):
    n = X.shape[1]
    p = np.exp(X.T.dot(theta))
    p = p / np.sum(p, axis=1).reshape(-1, 1)
    pred = np.argmax(p,axis=1)
    judge = (pred==y.reshape(n))
    accuracy = np.sum(judge)/float(n)
    return accuracy
