import numpy as np
from numpy import linalg, array
import numpy.random as random
from cla_utils.exercises3 import householder_qr, householder_solve
from cla_utils.exercises7 import solve_LUP
from cla_utils.exercises2 import GS_modified_R
from cla_utils.exercises5 import solve_R
from cla_utils.exercises8 import hessenberg



def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    count = 0
    if store_iterations:
        xlist = [list(x0)]
    while True:
        if count >= maxit:
            break
        w = A@x0
        x0 = w/np.linalg.norm(w)
        lambda0 = np.dot(x0,np.conj(A @ x0))
        r = A @ x0 - lambda0 * x0
        if store_iterations:
            xlist.append(list(x0))
        count+=1
        if np.linalg.norm(r) < tol:
            break
    if store_iterations:
        x = np.array(xlist).T
    if not store_iterations:
        x = x0
    
    return x, lambda0





def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    x0 = np.reshape(x0,(len(x0),1))
    count = 0
    if store_iterations:
        xlist = [list(x0)]
        llist = []
    while True:
        if count >= maxit:
            break
        A0 = 1.0*A
        #w = householder_solve(A-mu*np.eye(len(A)), x0) slower than numpy, change back after testing
        w = np.linalg.solve(A-mu*np.eye(len(A)), x0)
        x0 = w/np.linalg.norm(w)
        lambda0 = np.dot(x0.T,np.conj(A @ x0))[0][0]
        r = A @ x0 - lambda0 * x0
        if store_iterations:
            xlist.append(list(x0))
            llist.append(lambda0)
        count+=1
        if np.linalg.norm(r) < tol:
            break
    if store_iterations:
        x = np.array(xlist).T
        l = np.array(llist)
    if not store_iterations:
        x = x0
        l = lambda0
    
    return x, l


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    x0 = np.reshape(x0,(len(x0),1))
    lambda0 = np.dot(x0.T, np.conj(A @ x0))[0][0]
    count = 0
    if store_iterations:
        xlist = [list(x0)]
        llist = [lambda0]
    while True:
        if count >= maxit:
            break
        A0 = 1.0*A
        #w = householder_solve(A-lambda0*np.eye(len(A)), x0) slower than numpy, change back after testing
        w = np.linalg.solve(A-lambda0*np.eye(len(A)), x0)
        x0 = w/np.linalg.norm(w)
        lambda0 = np.dot(x0.T, np.conj(A @ x0))[0][0]
        r = A @ x0 - lambda0 * x0
        if store_iterations:
            xlist.append(list(x0))
            llist.append(lambda0)
        count+=1
        if np.linalg.norm(r) < tol:
            break
    if store_iterations:
        x = np.array(xlist).T
        l = np.array(llist)
    if not store_iterations:
        x = x0
        l = lambda0
    
    return x, l


def pure_QR(A, maxit, tol, symmetric=False, shifted=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    
    if symmetric:
        hessenberg(A)
        botdiag = []
    count = 0
    while True:
        m = A.shape[0]
        if count >= maxit:
            break
        #Q, R = householder_qr(A) slower than numpy, change back after testing
        if shifted:
            a = A[m-1,m-1]
            b = A[m-1,m-2]
            delta = (A[m-2,m-2]-A[m-1,m-1])/2
            mu = a - np.sign(delta)* b ** 2 / (abs(delta) + np.sqrt(delta**2 + b**2))
            #Q, R = householder_qr(A - mu*np.eye(m)) change back after testing
            Q, R = np.linalg.qr(A- mu*np.eye(m))
            A = R @ Q + mu * np.eye(m)
        if not shifted:
            #Q, R = householder_qr(A) change back after testing
            Q, R = np.linalg.qr(A)
            A = R @ Q
        count+=1
        #can use this convergence test if we know that we will converge to upper triangular as it speeds up process
        #if np.linalg.norm(np.tril(A,-1) < tol: 
            #break
        if symmetric:
            botdiag.append(abs(A[m-1,m-2]))

        if symmetric and abs(A[m-1,m-2]) < 10e-12:
            break
        if not symmetric and np.linalg.norm(Q @ R - R @ Q) < tol:
            break
    if symmetric:
        return A, np.array(botdiag)
    return A
