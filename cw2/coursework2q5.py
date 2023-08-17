import numpy as np
from cla_utils.exercises6 import LU_inplace
def LU_banded_inplace(A):
    m = A.shape[0]
    L = np.eye(m)
    p = np.where(A[0,:] == 0)[0][0]
    q = np.where(A[:,0] == 0)[0][0]
    opcount = 0
    for k in range(m-1):
        for j in range(k+1, min(k+q,m)):
            L[j, k] = A[j, k] / A[k, k]
            opcount += 1
            n = min(k+p, m)
            A[j, k:n] = A[j, k:n] - L[j, k] * A[k, k:n]
            opcount += 2 * (n - k)
    
    return L, A, opcount
count = []
for m in range(1000,10000, 1000):
    A = np.diag(np.random.randn(m),0) + np.diag(np.random.randn(m-1),1) + np.diag(np.random.randn(m-1),-1)
    count.append(LU_banded_inplace(A)[2])
print(count)
#this returns [4995, 9995, 14995, 19995, 24995, 29995, 34995, 39995, 44995] which we can see is clearly linear growth in m
count = []
for m in range(1000,10000, 1000):
    A = np.diag(np.random.randn(m),0) + np.diag(np.random.randn(m-1),1) + np.diag(np.random.randn(m-1),-1) +  np.diag(np.random.randn(m-2),-2) + np.diag(np.random.randn(m-2),2)
    count.append(LU_banded_inplace(A)[2])
print(count)
#this returns [13977, 27977, 41977, 55977, 69977, 83977, 97977, 111977, 125977] which we can see is clearly linear growth in m
count = []
for m in range(1000,10000, 1000):
    A = np.diag(np.random.randn(m),0) + np.diag(np.random.randn(m-1),1) + np.diag(np.random.randn(m-1),-1) +  np.diag(np.random.randn(m-2),-2)
    count.append(LU_banded_inplace(A)[2])
print(count)
#this returns [9985, 19985, 29985, 39985, 49985, 59985, 69985, 79985, 89985] which we can see is clearly linear growth in m
count = []
for m in range(1000,10000, 1000):
    A = np.diag(np.random.randn(m),0) + np.diag(np.random.randn(m-1),1) + np.diag(np.random.randn(m-1),-1) + np.diag(np.random.randn(m-2),2)
    count.append(LU_banded_inplace(A)[2])
print(count)
#this returns [6991, 13991, 20991, 27991, 34991, 41991, 48991, 55991, 62991] which we can see is clearly linear growth in m

