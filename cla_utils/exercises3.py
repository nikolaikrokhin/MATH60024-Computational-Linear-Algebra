import numpy as np
from numpy.lib.arraysetops import isin
import scipy.linalg as linalg
import pytest


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """
    def sign(n):
        if isinstance(n, float):
            if n == 0:
                return 1
            return n / np.linalg.norm(n)
        if isinstance(n, np.ndarray):
            if n[0] == 0:
                return 1
            return n[0] / np.linalg.norm(n[0])

    m, n = A.shape
    if kmax is None:
        kmax = n
    
    for k in range(kmax):
        x = A[k:m, k]
        try:
            
            v = sign(x) * np.linalg.norm(x) * np.eye(len(x))[:, 0] + x
            v = v / np.linalg.norm(v)
            A[k:m, k:n] = A[k:m, k:n] - 2 * np.outer(np.conj(v), (np.dot(v, A[k:m, k:n])))
            #print(A) 
        except IndexError:
            pass
        



    return A


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    m, n = A.shape
    if len(b.shape) == 2:
        k = b.shape[1]
    if len(b.shape) == 1:
        k = 1
    
    Ahat = np.empty((m, n + k),dtype=np.complex128)
    Ahat[:, :n] = A
    Ahat[:, n:] = b
    Rx = householder(Ahat)
    x = linalg.solve_triangular(Rx[:, :n], Rx[:, n:])
    

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    Ahat = np.empty((m, m + n),dtype=A.dtype)
    Ahat[:, :n] = A
    Ahat[:, n:] = np.eye(m)
    R, Q  = householder(Ahat)[:, :n], householder(Ahat)[:, n:].T

    return -Q, -R #just to stay consistent with the sign of Q, R from GS_modified 


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """
    m, n = A.shape
    Ahat = np.empty((m, m + n))
    Ahat[:, :n] = A
    Ahat[:, n:] = np.eye(m)
    Q, R = householder_qr(Ahat)
    R = R[:n, :n]
    Q = Q[:, :n]
    x = linalg.solve_triangular(R, b.dot(Q))


    return x

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
