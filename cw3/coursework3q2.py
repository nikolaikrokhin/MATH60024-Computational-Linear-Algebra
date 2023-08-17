import numpy as np
import pytest
from cla_utils.exercises9 import pure_QR


def makeA(n):
    return np.diag(np.ones(2*n-1),1) + np.diag(-np.ones(2*n-1),-1)
#observing A for different n
#print(np.around(pure_QR(makeA(2), maxit=10000, tol=1.0e-5),1))
#print(np.around(pure_QR(makeA(3), maxit=10000, tol=1.0e-5),1))
def pure_QRmod(A, maxit, tol):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """

    A0 = A
    count = 0
    while True:
        if count >= maxit:
            break
        #Q, R = householder_qr(A0) slower than numpy, change back after testing
        Q, R = np.linalg.qr(A0)
        #print("Q \n",Q)
        #print("R \n",R)
        
        A0 = R @ Q
        #print(f"A{count+1} \n", A0)
        count+=1
        #if np.linalg.norm(np.tril(A0,-1)) < tol:
            #break
    #if np.linalg.norm(np.tril(A0,-1)) > tol:
        #return "Not Converged"
    return A0
G = pure_QRmod(makeA(3), maxit=10000, tol=1.0e-5)

print(np.around(G,1))
print(np.around(np.linalg.eigvals(makeA(3)),3))
print(np.linalg.eigvals(G),1)
def Aevals(n):
    """
    Input: n
    Output: Eigenvalues for matrix A of size 2n x 2n
    """
    A = makeA(n)
    C = pure_QR(A, maxit=10000, tol=1.0e-5)
    topdiag = np.diag(C,1)
    etop = 1j*topdiag[::2]
    return np.concatenate((etop,-etop))
print(Aevals(3))

@pytest.mark.parametrize('m', [2, 9, 18])
def test_Aevals(m):
    np.random.seed(3213*m)
    A = makeA(m)
    evals = Aevals(m)
    for i in evals:
        assert(np.linalg.det(A - i*np.eye(2*m)) < 1.0e-5)


def makeB(n):
    return np.diag(np.array([2]*(2*n-1)),1) + np.diag(-np.ones(2*n-1),-1)

print(makeB(3))
print((np.around(pure_QRmod(makeB(3),maxit=10000,tol=1.0e-7),3)))
print((np.linalg.eigvals(makeB(3))))


def Bevals(n):
    """
    Input: n
    Output: Eigenvalues for matrix B of size 2n x 2n
    """
    B = makeB(n)
    C = pure_QR(B, maxit=10000, tol=1.0e-5)
    topdiag = np.diag(C,1)
    botdiag = np.diag(C,-1)
    etop = topdiag[::2]
    ebot = botdiag[::2]
    evals = np.sqrt(abs(etop * ebot))
    return 1j*np.concatenate((evals,-evals))
print(Bevals(3))


@pytest.mark.parametrize('m', [2, 9, 18])
def test_Bevals(m):
    np.random.seed(3213*m)
    B = makeB(m)
    evals = Bevals(m)
    for i in evals:
        assert(np.linalg.det(B - i*np.eye(2*m)) < 1.0e-5)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)



