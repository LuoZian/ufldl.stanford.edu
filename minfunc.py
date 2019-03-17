import numpy as np
def getfg(theta,X,y,head):
    m = X.shape[1]
    delta_y = np.matmul(theta.T,X)-y  #theta:14*1 delta_y:1*400
    f = 0.0025*0.5*np.matmul(delta_y,delta_y.T) #f 1*1
    if head + 10 <= m:
        g = 0.1 * np.matmul(X[:,head:head+10], delta_y.T[head:head+10,:])
        head = head+10
    else:
        sliceh =np.arange(0,head+10-m,1)
        slicer =np.arange(head,m,1)
        slice = np.concatenate((sliceh,slicer))
        g = 0.1 * np.matmul(X[:,slice], delta_y.T[slice,:])
        head = head+10-m               #g 14*1
    return f,g,head
def minfunc(theta,X,y):
    spot = 0
    for i in range(100000):
        fg = getfg(theta,X,y,spot)
        theta = theta -fg[1]*0.000005
        fnow = fg[0]
        spot = fg[2]
    return theta,float(fnow)  #用不用返回theta呢





