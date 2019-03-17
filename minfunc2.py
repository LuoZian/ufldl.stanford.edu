import numpy as np
import math
def h_theta(theta,x):
    a = 1+math.exp(-np.matmul(theta.T,x))
    return 1.0/a
def getfg(theta,X,y):
    h = [h_theta(theta, i.reshape(785, 1)) for i in X.T ]
    h = np.array(h).reshape(X.shape[1],1)-y.T
    g = np.matmul(X,h)
    return g
def minfunc(theta,X,y):
    for i in range(20):
       g = getfg(theta,X,y)
       theta = theta -g*0.0001
    return theta
def predict_y(theta,x):
    if np.matmul(theta.T,x)<0:
        return 0
    else:
        return 1