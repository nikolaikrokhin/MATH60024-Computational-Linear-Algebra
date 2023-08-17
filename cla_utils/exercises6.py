import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.
    :return Lk: an mxm dimensional numpy array.

    """
    k = m - len(lvec) - 1
    A = np.eye(m)
    A[k+1:,k] = lvec
    return A

def LU_inplace(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
                     
    m = len(A)
    for k in range(m-1):
        A[k+1:,k] = A[k+1:,k] / A[k, k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:].conj())
        
    



def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m = L.shape[0]
    k = b.shape[1]
    x = np.empty((m,k),dtype=L.dtype)
    for i in range(m):
        x[i] = (b[i] - L[i, :i].dot(x[:i]) ) / L[i,i]
    return x

    

    


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
    m = U.shape[0]
    k = b.shape[1]               
    x = np.empty((m,k),dtype=U.dtype)
    x[-1] = b[-1] / U[-1, -1]
    for i in range(m-2,-1,-1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x
                     
    


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
    m = A.shape[0]                 
    LU_inplace(A)
    U = np.triu(A)
    L = np.tril(A)
    np.fill_diagonal(L, 1)
    y = solve_L(L, np.eye(m))
    x = solve_U(U, y)
    return x

