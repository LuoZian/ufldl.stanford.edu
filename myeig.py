import numpy as np
def eig(A):
    n = A.shape[0]
    HT = np.identity(n)
    Epsilon = 0
    for i in range(n):
        for j in range(i):
            Epsilon += 2*A[i,j]*A[i,j]
    print(Epsilon)
    for i in range(2500):
        p, q = maxA(A)
        Epsilon -= 2 * A[p, q]* A[p, q]
        theta = np.arctan(2*A[p,q]/(A[p,p]-A[q,q]))/2
        c = np.cos(theta)
        s = np.sin(theta)
        for j in range(n):
            temp1 = HT[j,p]*c+HT[j,q]*s
            temp2 = -HT[j,p]*s+HT[j,q]*c
            HT[j,p] = temp1
            HT[j,q] = temp2
            if j !=p and j!=q:
                temp1 = A[p,j]*c+A[q,j]*s
                temp2 = A[q,j]*c-A[p,j]*s
                A[p,j] = A[j,p] = temp1
                A[q,j] = A[j,q] = temp2
        temp1 = A[p,p]*c*c+A[q,q]*s*s+2*A[p,q]*s*c
        temp2 = A[p,p]*s*s+A[q,q]*c*c-2*A[p,q]*s*c
        A[p,p] = temp1
        A[q,q] = temp2
        A[p,q] = A[q,p] = 0
    list1 = [-A[i,i] for i in range(n)]
    temp = np.array(list1)
    slic = np.argsort(temp)
    Ans = np.zeros((n,n))
    for i in range(n):
        Ans[i,i] = A[slic[i],slic[i]]
    HT = HT[:,slic]

    print(Epsilon)
    return Ans,HT
def maxA(A):
    n = A.shape[0]
    maxA = -1
    p = -1
    q = -1
    for i in range(n):
        for j in range(i):
            if(A[i,j]*A[i,j]>maxA):
                maxA = A[i,j]*A[i,j]
                p = j
                q = i
    return p,q
def SVD(A):
    r = np.linalg.matrix_rank(A)
    sigma,U = eig(A.dot(A.T))
    sigma = sigma[0:r,0:r]
    sigma = np.power(sigma,0.5)
    U = U[:,0:r]
    for i in range(r):
        temp = U[:,i:i+1].T.dot(A)/sigma[i,i]
        if(i == 0):
            VT = temp
        else:
            VT = np.concatenate((VT,temp),axis=0)
    return U,sigma,VT
