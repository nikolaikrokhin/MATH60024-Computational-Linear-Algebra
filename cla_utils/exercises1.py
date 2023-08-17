import numpy as np
import timeit
import numpy.random as random
import pytest

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """

    b = np.zeros(len(A))
    for i in range(len(A)):
        sum = 0
        for j in range(len(x)):
            sum += A[i][j] * x[j]
        b[i] = sum
    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """

    b = np.zeros(len(A))
    for i in range(len(x)):
        b += x[i] * A[:, i]
    return b


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    return np.outer(u1, np.conjugate(v1)) + np.outer(u2, np.conjugate(v2))

    


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """

    alpha = -1 / (1 + np.vdot(v,u))
    return np.identity(len(u)) + alpha * np.outer(u, v.conj())

u, v = np.random.randn(400) * complex(1, 0) + np.random.randn(400) * complex(0, 1), np.random.randn(400) * complex(1, 0) + np.random.randn(400) * complex(0, 1)
A1 = np.identity(400) + np.outer(u, v.conj())
def timable_rank1pert_inv():
    A = rank1pert_inv(u, v)
def timable_numpy_inv():
    A = np.linalg.inv(A1)

def time_rank1pert_inv():
    print("Timing for my function")
    print(timeit.Timer(timable_rank1pert_inv).timeit(number=1))
    print("Timing for np inverse")
    print(timeit.Timer(timable_numpy_inv).timeit(number=1))

time_rank1pert_inv()
#we see that our trick is much faster (about 10 times) than numpy inverse

def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """

    n = len(Ahat)
    Ahat = Ahat.T
    zr, zi = np.zeros(n), np.zeros(n)
    for j in range(n):
        B, C = np.zeros(n), np.zeros(n)
        B[:j] = Ahat[:j, j]
        B[j:] = Ahat[j, j:]
        C[:j] = Ahat[j, :j]
        C[j+1:] = -Ahat[j+1:, j]
        zr += B * xr[j] - C * xi[j]
        zi += B * xi[j] + C * xr[j]
    return zr, zi

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)