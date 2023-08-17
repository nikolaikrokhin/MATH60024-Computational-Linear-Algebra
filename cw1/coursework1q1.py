import numpy as np
import sys, os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..', 'cla_utils')))
import cla_utils
import pytest

C = np.loadtxt('cw1/C.dat', delimiter=',')

def compressC(A):
    """
    Given the matrix C, compute the compression of C by QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    only the first three columns of Q and first three rows of R.

    :param A: mxn numpy array

    :return Q, R: mx3, 3xn numpy array
    """

    m, n = A.shape
    R = np.zeros((n,n))
    for i in range(3):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = (A[:, i].conj().T).dot(A[:, i+1:])
        A[:, i+1:] = A[:, i+1:] - np.outer(A[:, i],R[i, i+1:])
    return A[:, :3], R[:3, :]

@pytest.mark.parametrize('m, n', [(1, 1)])
def test_compressC(m, n):
    C = np.loadtxt('cw1/C.dat', delimiter=',')
    C0 = 1.0*C
    Q, R = compressC(C)
    assert(np.allclose(C0, Q @ R))  # check product of Q and R from compression is equal to C

Q, R = cla_utils.householder_qr(C)
#writing R to file
a_file = open("file1.txt", "w")
np.savetxt(a_file, R)
a_file.close()








