import numpy as np
import timeit
import pytest


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    r = v
    u = np.zeros(len(Q[0]))
    for i in range(len(Q[0])):
        r -= np.vdot(Q[:, i], v) * Q[:, i]
        u[i] = np.vdot(Q[:, i], v)
        

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    x = np.asmatrix(Q).H @ b

    return x
Q1, Q2, Q3 = np.random.rand(100, 100), np.random.rand(200, 200), np.random.rand(400, 400)
b1, b2, b3 = np.random.rand(100), np.random.rand(200), np.random.rand(400)
def timeable_solveQ1():
    A = solveQ(Q1,b1)
def timeable_solveQ2():
    A = solveQ(Q2,b2)
def timeable_solveQ3():
    A = solveQ(Q3,b3)
def timeable_solveQ1_numpy():
    A = np.linalg.solve(Q1,b1)
def timeable_solveQ2_numpy():
    A = np.linalg.solve(Q2,b2)
def timeable_solveQ3_numpy():
    A = np.linalg.solve(Q3,b3)  

def time_solveQ():
    print("Timing for my function of size 100")
    print(timeit.Timer(timeable_solveQ1).timeit(number=1))
    print("Timing for np solve of size 100")
    print(timeit.Timer(timeable_solveQ1_numpy).timeit(number=1))
    print("Timing for my function of size 200")
    print(timeit.Timer(timeable_solveQ2).timeit(number=1))
    print("Timing for np solve of size 200")
    print(timeit.Timer(timeable_solveQ2_numpy).timeit(number=1))
    print("Timing for my function of size 400")
    print(timeit.Timer(timeable_solveQ3).timeit(number=1))
    print("Timing for np solve of size 400")
    print(timeit.Timer(timeable_solveQ3_numpy).timeit(number=1))

time_solveQ()
#we notice for small size matrices like 100x100 we have np.solve faster but as dimension increases our implemented method becomes much faster
    


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    P = np.asmatrix(Q) @ np.asmatrix(Q).H

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """

    n = V.shape[1]
    Q = np.linalg.qr(V, mode='complete')[0]
    return Q[:, n:]


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    n = A.shape[1]
    R = np.zeros((n,n))
    for i in range(n):
        R[:i, i] = (A[:, :i].conj().T).dot(A[:, i])
        A[:, i] =  A[:, i] - A[:, :i].dot(np.conj(R[:i, i]))
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        
    return R
def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.zeros((n,n))
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = (A[:, i].conj().T).dot(A[:, i+1:])
        A[:, i+1:] = A[:, i+1:] - np.outer(A[:, i],R[i, i+1:])

        




    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    n = A.shape[1]
    R = np.eye(n,dtype=A.dtype)
    R[k, k] = 1 / np.linalg.norm(A[:, k])
    R[k, k+1:] = R[k, k+1:] - (A[:, k].conj().T).dot(A[:, k+1:])*R[k, k]

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, np.conj(Rk))
        R[:,:] = np.dot(R, np.conj(Rk))
    R = np.linalg.inv(R)
    return A, R

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
