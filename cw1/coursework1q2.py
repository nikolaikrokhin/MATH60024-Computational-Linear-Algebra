import numpy as np
import sys, os
import scipy
sys.path.append(os.path.abspath(os.path.join('..', 'cla_utils')))
import cla_utils

#setting up matrix equation
x = np.arange(0., 1.0001, 1./51)
A = np.vander(x, 13, increasing=True)
A0 = 1.0*A
A1 = 1.0*A
A2 = 1.0*A
A3 = 1.0*A
A4 = 1.0*A
A5 = 1.0*A
b = np.array([1] + [0] * 49 + [1] + [0])

#GS_classical
def GS_classical_solve(A, b):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place then solving Ax = b.

    :param A: mxn numpy array, b:mx1 numpy array

    :return x: nx1 numpy array
    """
    n = A.shape[1]
    R = np.zeros((n,n))
    for i in range(n):
        R[:i, i] = (A[:, :i].conj().T).dot(A[:, i])
        A[:, i] =  A[:, i] - A[:, :i].dot(R[:i, i])
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
    
        
    return cla_utils.linalg.solve_triangular(R, b.dot(A))

cfGS = GS_classical_solve(A0, b)

#GS_modified
def GS_modified_solve(A, b):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and solving Ax = b

    :param A: mxn numpy array, b:mx1 numpy array

    :return x: nx1 numpy array
    """

    m, n = A.shape
    R = np.zeros((n,n))
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = (A[:, i].conj().T).dot(A[:, i+1:])
        A[:, i+1:] = A[:, i+1:] - np.outer(A[:, i],R[i, i+1:])
    
    return cla_utils.linalg.solve_triangular(R, b.dot(A))
cfGSM = GS_modified_solve(A1, b)
#Householder
cfHH = cla_utils.householder_ls(A2, b)

#writing to files
a_file = open("q2file1.txt", "w")
np.savetxt(a_file, cfGS)
a_file.close()
a_file = open("q2file2.txt", "w")
np.savetxt(a_file, cfGSM)
a_file.close()
a_file = open("q2file3.txt", "w")
np.savetxt(a_file, cfHH)
a_file.close()

def GS_classical_QR(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning Q, R.

    :param A: mxn numpy array

    :return Q, R: mxn, nxn numpy arrays
    """
    n = A.shape[1]
    R = np.zeros((n,n))
    for i in range(n):
        R[:i, i] = (A[:, :i].conj().T).dot(A[:, i])
        A[:, i] =  A[:, i] - A[:, :i].dot(R[:i, i])
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        
    return A, R

def GS_modified_QR(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    Q, R.

    :param A: mxn numpy array

    :return Q, R: mxn, nxn numpy array
    """

    m, n = A.shape
    R = np.zeros((n,n))
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = (A[:, i].conj().T).dot(A[:, i+1:])
        A[:, i+1:] = A[:, i+1:] - np.outer(A[:, i],R[i, i+1:])

    return A, R

GSQ, GSR = GS_classical_QR(A3)
GSMQ, GSMR = GS_modified_QR(A4)
HHQ, HHR = cla_utils.householder_qr(A5)
#writing files
a_file = open("q2file4.txt", "w")
np.savetxt(a_file, GSQ)
a_file.close()
a_file = open("q2file5.txt", "w")
np.savetxt(a_file, GSR)
a_file.close()
a_file = open("q2file6.txt", "w")
np.savetxt(a_file, GSMQ)
a_file.close()
a_file = open("q2file7.txt", "w")
np.savetxt(a_file, GSMR)
a_file.close()
a_file = open("q2file8.txt", "w")
np.savetxt(a_file, HHQ)
a_file.close()
a_file = open("q2file9.txt", "w")
np.savetxt(a_file, HHR)
a_file.close()