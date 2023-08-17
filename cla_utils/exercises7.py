import numpy as np

from cla_utils.exercises6 import solve_L, solve_U

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    p0 = p*1.0
    p[i], p[j] = p0[j], p0[i]
    

    


def LUP_inplace(A):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
                     
    m = A.shape[0]
    P = np.arange(m)
    for k in range(m-1):
        i = np.argmax(np.abs(A[k:, k])) + k
        perm(A, i, k)
        perm(P, i, k)
        A[k+1:,k] = A[k+1:,k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:].conj())
    return P


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    m = len(b)
    b = np.reshape(b, (m,1))
    P = LUP_inplace(A)
    U = np.triu(A)
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    y = solve_L(L, b[P])
    x = solve_U(U, y)
    return x[:,0]

                     
    

def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
    m = A.shape[0]               
    P = LUP_inplace(A) 
    sgn = 1  #initialise sign of permutation
    for i in range(m-1):  #iterate until return to identity
        if P[i] != i: #check if position not correct in permuation
            sgn *= -1 #adjust sign if swap needed
            l = min(range(i,m), key=P.__getitem__) #search pos with min number of swaps ahead of current pos
            P[i],P[l] = P[l],P[i] #swap current position with possible min number of swap position
    

    return np.diag(A).prod() * sgn #return det P * det U (since det L = 1)

    

