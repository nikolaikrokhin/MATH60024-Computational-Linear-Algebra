import numpy as np
from numpy import dtype, random
from numpy.core.fromnumeric import size
import pytest
from cla_utils.exercises6 import LU_inplace, inverse_LU
def makeA(n):
    A = np.eye(4*n + 1)
    eps = np.random.uniform(0.0, 0.1)
    for k in range(n):
        B = np.zeros((4*n+1, 4*n+1))
        B[4*k:4*k + 5, 4*k:4*k + 5] = np.random.uniform(low = 0, high = 1.0, size = (5,5))
        A = A + eps * B
    return(A)

x = makeA(2)
a_file = open("x.txt", "w")
np.savetxt(a_file, x)
a_file.close()
LU_inplace(x)
a_file = open("upper.txt", "w")
np.savetxt(a_file, np.triu(x))
a_file.close()
w=np.tril(x)
np.fill_diagonal(w, 1)
a_file = open("lower.txt", "w")
np.savetxt(a_file, w)
a_file.close()

def LU_block_inplace(A):
    m = A.shape[0]
    L = np.eye(m)
    ind = [5 + 4 * int(np.floor(k/4)) for k in range(m-1)]
    for k in range(m-1):
        z = ind[k]
        for j in range(k+1, z):
            L[j, k] = A[j, k] / A[k, k]
            A[j, k:z] = A[j, k:z] - L[j, k] * A[k, k:z]
    return L, A

y = makeA(7)
y1 = 1.0*y
y2 = 1.0*y
y3 = 1.0*y
LU_inplace(y2)
a_file = open("LU_inplace.txt", "w")
np.savetxt(a_file, np.triu(y2))
a_file.close()
a_file = open("LU_block_inplace.txt", "w")
np.savetxt(a_file, LU_block_inplace(y1)[1])
a_file.close()
#print(LU_block_inplace(y3)[1] - np.triu(y2))

@pytest.mark.parametrize('m', [7, 131, 48])
def test_LU_block_inplace(m):
    np.random.seed(854*m)
    A = makeA(m)
    A1 = 1.0*A
    A2 = 1.0*A
    LU_inplace(A1)
    U1 = np.triu(A1)
    L1 = np.tril(A1)
    np.fill_diagonal(L1, 1)
    L2, U2 = LU_block_inplace(A)
    err1 = U1 - U2
    err2 = L1 - L2
    assert(np.linalg.norm(err1) < 1.0e-6 and np.linalg.norm(err2) < 1.0e-6 and np.linalg.norm(np.dot(L2,U2) - A2) < 1.0e-6)

def LU_block_inplace_refined(A):
    m = A.shape[0]
    for k in range(int((m-1)/4)):
        LU_inplace(A[4*k:4*k+5,4*k:4*k+5])
        print((4*k,4*k+5))
#print(LU_block_inplace_refined(makeA(4)))

@pytest.mark.parametrize('m', [20, 204, 18])
def test_LU_block_inplace_refined(m):
    np.random.seed(8564*m)
    A = makeA(m)
    A0 = 1.0*A
    LU_block_inplace_refined(A)
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)

def maketridiag(A):
    m = A.shape[0]
    Atilde = []
    for i in range(int((m-1)/4)):
        for j in range(int((m-1)/4)):
            if i == 0 and j == 0:
                Atilde.append(A[0:5, 0:5])
                #print((i,j))
                #print("one")
            if i == 0 and j !=0:
                Atilde.append(A[0:5, 4 * (j) + 1: 4 * (j+1) + 1])
                #print((i,j))
                #print("two")
            if i !=0 and j == 0:
                Atilde.append(A[4 * (i) + 1: 4 * (i+1) + 1, 0:5])
                #print((i,j))
                #print("three")
            if i !=0 and j != 0:
                Atilde.append(A[4 * (i - 1) + 1: 4 * i + 1, 4 * (j - 1) + 1: 4 * j + 1])
                #print((i,j))
                #print("four")
            
            #print(Atilde[-1])    
    Atilde = np.asarray(Atilde, dtype='object')
    Atilde = Atilde.reshape(int((m-1)/4),int((m-1)/4))
    return Atilde

#myA = makeA(4)
#At = maketridiag(myA)
#print(At)

def sizeb(A,b):
    m = A.shape[0]
    c = [b[0:5]]
    for i in range(1, int((m-1)/4)):
        c.append(b[4*i + 1: 4*(i+1)+1])
    c = np.asarray(c, dtype='object')
    c = c.reshape(int((m-1)/4),1)
    return c


def LU_tridiag_solve(A, b):
    Atilde = maketridiag(A)
    m = Atilde.shape[0]
    L = maketridiag(np.eye(A.shape[0]))
    for k in range(m-1):
        for j in range(k+1, min(k+2,m)):
            L[j, k] = Atilde[j, k] @ inverse_LU(Atilde[k, k])
            
            for n in range(k,min(k+3, m)):
                Atilde[j, n] = Atilde[j, n] - L[j, k] @ Atilde[k, n] #we have now created L and U
            
    c = sizeb(A, b)
    y = sizeb(A, np.zeros(A.shape[0]))
    x = sizeb(A, np.zeros(A.shape[0]))
    y[0] = c[0]
    for i in range(1,m):
        y[i][0] = c[i][0] - L[i,i-1] @ y[i-1][0] #implementing forward subst
    x[-1][0] = inverse_LU(Atilde[-1, -1]) @ y[-1][0]
    for i in range(m-2,-1,-1):
        x[i][0] = inverse_LU(Atilde[i,i]) @ (y[i][0] - Atilde[i, i+1] @ x[i+1][0]) #implementing backward subst
    final = []  #concatenating the values back to the original x
    for v in x:
        final = final + list(v[0])
    return np.asarray(final)


@pytest.mark.parametrize('m', [42, 2, 91])
def test_LU_tridiag_solve(m):
    random.seed(854*m)
    A = makeA(m)
    x = np.random.randn(A.shape[0])
    b = A @ x
    myx = LU_tridiag_solve(A, b)
    print(myx)
    print(x)
    err = x - myx
    assert(np.linalg.norm(x- myx) < 10e-2)


    
    
    



x = makeA(3)
b = np.random.randn(13,1)
print(LU_tridiag_solve(x, b))
#print(LU_tridiag_solve(x,1))
#print("-----")
#print(LU_tridiag_solve(x, 1)[0])
print("-----")
#print(LU_tridiag_solve(x, 1)[1])










