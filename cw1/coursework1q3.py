import numpy as np
import sys, os
from numpy.lib.function_base import iterable
from numpy.linalg import norm
import pandas as pd

from cla_utils.exercises3 import householder_qr
sys.path.append(os.path.abspath(os.path.join('..', 'cla_utils')))
import cla_utils
import pytest
import scipy

def Rv_method(A):
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
            return np.sign(n)
        if isinstance(n, np.ndarray):
            if n[0] == 0:
                return 1
            return np.sign(n[0])

    m, n = A.shape
    for k in range(n):
        x = A[k:m, k]
        try:
            
            v = sign(x) * np.linalg.norm(x) * np.eye(len(x))[:, 0] + x
            #print(v)
            norm = np.linalg.norm(v)
            v = v / norm
            #print(v)
            A[k:m, k:n] = A[k:m, k:n] - 2 * np.outer(v, (np.dot(v, A[k:m, k:n])))
            #print(A) 
            
            A[k+1:, k] = v[1:] * norm
            #print(A)
        except IndexError:
            pass
    return A

        
C = np.random.randn(5,3) * 10
b = np.random.randn(5) * 10
C0 = 1.0 * C
C1 = 1.0 * C
b0 = 1.0*b
b1 = 1.0*b
#print(Rv_method(C0))
#print("-----------------")
#Q, R = cla_utils.householder_qr(C0)
#print(Q.T @ b0)
#print("---------------")
def createQ(Rv):
    #create Q
    m, n = Rv.shape
    Q = np.eye(m)
    for i in range(n):
        x0 = -np.sign(Rv[i, i]) * np.sqrt(Rv[i, i] ** 2 - np.dot(Rv[i+1:, i], Rv[i+1:, i]))
        v0 = -np.sign(Rv[i, i]) * abs(Rv[i, i]) + x0
        v = np.concatenate((np.array([v0]), Rv[i+1:, i]))
        v = v / np.linalg.norm(v)
        F = np.eye(v.shape[0]) - 2 * np.outer(v, v)
        if i == 0:
            Q_k = F
        else:
            Q_k = scipy.linalg.block_diag(np.eye(i), F)
        Q = Q_k @ Q 
    return Q.T

@pytest.mark.parametrize('m, n', [(20, 9), (41, 13), (87, 19), (34, 11)])
def test_Rv_method(m, n):
    np.random.seed(1298*m + 53*n)
    A = np.random.randn(m, n)
    A0 = 1.0*A  # make a deep copy
    A1 = 1.0*A
    Rv = Rv_method(A1)
    assert(np.allclose(A, createQ(Rv) @ np.triu(Rv)))  # check product of Q and R from Rv is equal to A

def Qstarb(Rv, b):
    m, n = Rv.shape
    for i in range(n):
        x0 = -np.sign(Rv[i, i]) * np.sqrt(Rv[i, i] ** 2 - np.dot(Rv[i+1:, i], Rv[i+1:, i]))
        v0 = -np.sign(Rv[i, i]) * abs(Rv[i, i]) + x0
        v = np.concatenate((np.array([v0]), Rv[i+1:, i]))
        v = v / np.linalg.norm(v)
        F = np.eye(v.shape[0]) - 2 * np.outer(v, v)
        b[i:] = F @ b[i:] 

    return b


@pytest.mark.parametrize('m, n', [(20, 9), (41, 13), (87, 19), (34, 11)])
def test_Qstarb(m, n):
    np.random.seed(1798*m + 523*n)
    A = np.random.randn(m, n)
    x0 = np.random.randn(n)
    b = np.dot(A, x0)
    b1 = 1.0*b
    A0 = 1.0*A  # make a deep copy
    A1 = 1.0*A
    Q, R = cla_utils.householder_qr(A0)
    starb = Q.T @ b
    qstarb = Qstarb(Rv_method(A1), b1)
    assert(np.allclose(qstarb, starb))
    
def Rv_ls(A, b):
    m, n = A.shape
    Rv = Rv_method(A)
    R = np.triu(Rv)[:n, :]
    b = Qstarb(Rv, b)
    b = b[:n] 
    return scipy.linalg.solve_triangular(R, b)
    
    

@pytest.mark.parametrize('m, n', [(20, 9), (41, 13), (87, 19), (34, 11)])
def test_Rv_ls(m, n):
    np.random.seed(1798*m + 523*n)
    A = np.random.randn(m, n)
    x0 = np.random.randn(n)
    b = np.dot(A, x0)
    b1 = 1.0*b
    A0 = 1.0*A  # make a deep copy
    A1 = 1.0*A
    assert(np.allclose(Rv_ls(A1, b1), cla_utils.householder_ls(A0, b)))