from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from myeig import SVD
img = io.imread('butterfly.bmp',as_gray=True)
A = img.dot(img.T)
#U,sigma,VT= np.linalg.svd(img)
U,sigma,VT= SVD(img)
S = sigma
#S = np.zeros((243,243))
#for i in range(243):
#    S[i,i]=sigma[i]
A=np.zeros((243,437))
for i in range(0,243,250):
    end = i+25
    if(end>243):
        end = 243
    A=A+U[:,i:end].dot(S[i:end,i:end]).dot(VT[i:end,:])
    plt.imshow(A,cmap = plt.cm.gray)
    plt.show()