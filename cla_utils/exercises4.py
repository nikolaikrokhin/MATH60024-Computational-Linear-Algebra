import numpy as np
import pytest


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    w, v = np.linalg.eig(A.T @ A)
    o2norm = np.sqrt(max(w))
    return o2norm

@pytest.mark.parametrize('m, n', [(20, 9), (41, 75), (7, 2), (34, 52)])
def test_operator_inequality(m, n):
    np.random.seed(1298*m + 53*n)
    A = np.random.randn(m, n)
    x = np.random.randn(n)
    assert(np.linalg.norm(A @ x) <= operator_2_norm(A) * np.linalg.norm(x))  
#check
@pytest.mark.parametrize('l, m, n', [(12, 20, 9), (3, 41, 75), (5, 7, 2), (32, 34, 52)])
def test_matrix_inequality_theorem(l, m, n):
    np.random.seed(1298*l + 53*m + 3*n)
    A = np.random.randn(l, m)
    B = np.random.randn(m, n)
    
    assert(operator_2_norm(A @ B) <= operator_2_norm(A) * operator_2_norm(B))  


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    w, v = np.linalg.eig(A.T @ A)
    op2_norm_inv = np.sqrt(1/np.min(np.where(w>0, w, np.inf)))
    ncond = operator_2_norm(A) * op2_norm_inv

    return ncond
