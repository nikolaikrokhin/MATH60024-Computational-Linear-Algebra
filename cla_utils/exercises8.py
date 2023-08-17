import numpy as np
import matplotlib.pyplot as plt

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    def sign(n):
        if isinstance(n, float):
            if n == 0:
                return 1
            return np.sign(n)
        if isinstance(n, np.ndarray):
            if n[0] == 0:
                return 1
            return np.sign(n[0])

    m, n = A.shape
    x = A[:m, 0]
    v = sign(x) * np.linalg.norm(x) * np.eye(len(x))[:, 0] + x
    v = v / np.linalg.norm(v)
    A[:m, :m] = A[:m, :m] - 2 * np.outer(v, (np.dot(v, A[:m, :m])))
    A[:m, :m] = A[:m,:m] - 2 * A[:m, :m] @ np.outer(v, v)
    return A    
    


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """
    def sign(n):
        if isinstance(n, float):
            if n == 0:
                return 1
            return np.sign(n)
        if isinstance(n, np.ndarray):
            if n[0] == 0:
                return 1
            return np.sign(n[0])
    m, n = A.shape
    for k in range(m-2):
        x = A[k+1:m, k]
        v = sign(x) * np.linalg.norm(x) * np.eye(len(x))[:, 0] + x
        v = v / np.linalg.norm(v)
        A[k+1:m, k:m] = A[k+1:m, k:m] - 2 * np.outer(v, (np.dot(v, A[k+1:m, k:m])))
        #print(A)
        A[:m, k+1:m] = A[:m, k+1:m] - 2 * A[:m, k+1:m] @ np.outer(v, v)
        #print(A)
        #check exploitation



def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    def sign(n):
        if isinstance(n, float):
            if n == 0:
                return 1
            return np.sign(n)
        if isinstance(n, np.ndarray):
            if n[0] == 0:
                return 1
            return np.sign(n[0])
    m, n = A.shape
    Q = np.eye(m)
    for k in range(m-2):
        x = A[k+1:m, k]
        v = sign(x) * np.linalg.norm(x) * np.eye(len(x))[:, 0] + x
        v = v / np.linalg.norm(v)
        A[k+1:m, k:m] = A[k+1:m, k:m] - 2 * np.outer(v, (np.dot(v, A[k+1:m, k:m])))
        A[:m, k+1:m] = A[:m, k+1:m] - 2 * A[:m, k+1:m] @ np.outer(v, v)
        Q[:m, k+1:m] = Q[:m, k+1:m] - 2 * Q[:m, k+1:m] @ np.outer(v, v)
    return Q
    

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    return Q @ hessenberg_ev(A)

