import numpy as np
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
A = np.zeros((5,5)) + np.diag([1,4,6,2],-1) + np.diag([1,4,6,2],1) + np.diag([-3,5,-1,7,4],0)
print(A)
L, U, _ = LU_banded_inplace(A)
print(L @ U)
print(A)
