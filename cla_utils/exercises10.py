import numpy as np
from numpy.linalg import norm
import scipy
import numpy.random as random
from cla_utils.exercises3 import householder_ls
from cla_utils.exercises5 import solve_R
#from cw3.coursework3q4 import H_apply


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    Q = b / np.linalg.norm(b)
    Q = np.reshape(Q, (len(Q),1))
    H = np.zeros((k+1, k), dtype=b.dtype)
    for n in range(k):
        v = A @ Q[:, n]
        H[:n+1, n] = np.conj(Q.T) @ v
        v -= Q @ H[:n+1, n]
        H[n+1, n] = np.linalg.norm(v)
        v /= np.linalg.norm(v)
        v = np.reshape(v, (len(v),1))
        Q = np.concatenate([Q, v], axis=1)
    return Q, H
def GMRES(A ,b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False,matfunc=None,precondfunc=None):
    """
    USING GIVENS ROTATION!
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    def _grotate(x, n):
        t = np.arctan(x[n+1] / x[n])
        G = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        if n == 0:
            G = np.asarray(scipy.linalg.block_diag(np.eye(n), G))
            return G
        if n != 0:
            return np.cos(t), np.sin(t), G @ x[-2:]
        

    if x0 is None:
        x0 = b
    
    m = len(b)
    if precondfunc is None:
        normb = np.linalg.norm(b)
        Q = b / normb
    if precondfunc is not None:
        btilde = precondfunc(b)
        normbt = np.linalg.norm(btilde)
        Q = btilde / normbt
    Q = np.reshape(Q, (len(Q),1))
    H = np.zeros((m+1, m), dtype=b.dtype)
    rnormlist = []
    if return_residuals:
        r = np.zeros(m,m)
        
    count = 0
    ind = 0
    for n in range(m):
        if count >= maxit:
            break
        count+=1
        ind+=1
        if matfunc is not None:
            if precondfunc is None:
                v = matfunc(Q[:, n])
            if precondfunc is not None:
                v = precondfunc(matfunc(Q[:, n]))
        if matfunc is None:
            if precondfunc is None:
                v = A @ Q[:, n]
            if precondfunc is not None:
                v = precondfunc(A @ Q[:, n])
        
        H[:n+1, n] = np.conj(Q.T) @ v
        v -= Q @ H[:n+1, n]
        H[n+1, n] = np.linalg.norm(v)
        v /= np.linalg.norm(v)
        v = np.reshape(v, (len(v),1))
        Hn = H[:n+2,:n+1]
        if n == 0:
            G = _grotate(np.copy(Hn[:, -1]), n)
            Rg = G @ Hn[:, -1]
            Rg = np.reshape(Rg, (len(Rg),1))
            G = np.conj(G.T)
        if n != 0:
            G = np.asarray(scipy.linalg.block_diag(np.conj(G.T), np.array([1])))
            x = G @ Hn[:, -1]
            cos, sin, endx = _grotate(np.copy(x), n)
            Rg = np.concatenate([Rg, np.zeros([1, Rg.shape[1]])], axis=0)
            oRg = np.reshape(np.concatenate([x[:-2], endx]), (len(np.concatenate([x[:-2], endx])),1))
            Rg = np.concatenate([Rg, oRg], axis=1)
            nrow = np.copy(G[n, :])
            nrowp1 = np.copy(G[n+1, :])
            G[n+1, :] = -sin * nrow + cos * nrowp1
            G[n, :] = cos * nrow + sin * nrowp1
            G = np.conj(G.T)
        if precondfunc is None: 
            y = solve_R(Rg[:-1, :], np.conj(G[:, :-1].T) @ (normb * np.eye(n+2)[:, 0]))
            res = Hn @ y - normb * np.eye(Hn.shape[0])[:,0]
        if precondfunc is not None:
            y = solve_R(Rg[:-1, :], np.conj(G[:, :-1].T) @ (normbt * np.eye(n+2)[:, 0]))
            res = Hn @ y - normbt * np.eye(Hn.shape[0])[:,0]

        rnorm = np.linalg.norm(res)
        x = Q @ y
        Q = np.concatenate([Q, v], axis=1)
        if return_residual_norms:
            rnormlist.append(rnorm)
        if return_residuals:
            r[:count+2, count] = res
        if rnorm < tol:
            break
    if rnorm > tol:
        count = -2
    if return_residual_norms and return_residuals: 
        return x, count + 1, np.array(rnormlist), r[:,:ind+1]
    if return_residual_norms and not return_residuals: 
        return x, count + 1, np.array(rnormlist)
    if not return_residual_norms and return_residuals: 
        return x, count + 1, r[:,:ind+1]
    if not return_residual_norms and not return_residuals: 
        return x, count + 1
    return Q, H
    
#using householder reflections
#def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
#          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    """
    def _hhreflect(x, k):
        def sign(n):
            if isinstance(n, float):
                if n == 0:
                    return 1
                return n / np.linalg.norm(n)
            if isinstance(n, np.ndarray):
                if n[0] == 0:
                    return 1
                return n[0] / np.linalg.norm(n[0])
        normx = np.linalg.norm(x)
        v = sign(x[0])*normx*(np.eye(x.shape[0])[:, 0]) + x
        v = v / np.linalg.norm(v)
        F = np.eye(x.shape[0]) - 2 * np.outer(v, np.conj(v))
        F = np.asarray(scipy.linalg.block_diag(np.eye(k), F))

        return F, sign(x[0]), normx

    if x0 is None:
        x0 = b
    m = A.shape[0]
    normb = np.linalg.norm(b)
    Q = b / normb
    Q = np.reshape(Q, (len(Q),1))
    H = np.zeros((m+1, m), dtype=A.dtype)
    rnormlist = []
    if return_residuals:
        r = np.zeros(m,m)
        
    count = 0
    ind = 0
    for n in range(m):
        if count >= maxit:
            break
        count+=1
        ind+=1
        v = A @ Q[:, n]
        H[:n+1, n] = np.conj(Q.T) @ v
        v -= Q @ H[:n+1, n]
        H[n+1, n] = np.linalg.norm(v)
        v /= np.linalg.norm(v)
        v = np.reshape(v, (len(v),1))
        Hn = H[:n+2,:n+1]
        if n == 0:
            Qh, sn, normx = _hhreflect(np.copy(Hn[:, -1]), n)
            Rh = np.array([[-sn*normx], [0]])
        if n != 0:
            Qh = np.asarray(scipy.linalg.block_diag(Qh, np.array([1])))
            h = np.conj(Qh.T) @ Hn[:, -1]
            F, sn, normx = _hhreflect(np.copy(h[-2:]), n)
            Qh = Qh @ F
            oRh = np.concatenate([h[:-2], np.array([-sn*normx])])
            oRh = np.reshape(oRh,(len(oRh),1))
            Rh = np.concatenate([Rh, oRh], axis=1)
            Rh = np.concatenate([Rh, np.zeros([1, Rh.shape[1]])], axis=0)
        y = solve_R(Rh[:-1, :], np.conj(Qh[:, :-1].T) @ (normb * np.eye(n+2)[:, 0]))
        res = Hn @ y - normb * np.eye(Hn.shape[0])[:,0]
        rnorm = np.linalg.norm(res)
        x = Q @ y
        Q = np.concatenate([Q, v], axis=1)
        if return_residual_norms:
            rnormlist.append(rnorm)
        if return_residuals:
            r[:count+2, count] = res
        if rnorm < tol:
            break
    if rnorm > tol:
        count = -2
    if return_residual_norms and return_residuals: 
        return x, count + 1, np.array(rnormlist), r[:,:ind+1]
    if return_residual_norms and not return_residuals: 
        return x, count + 1, np.array(rnormlist)
    if not return_residual_norms and return_residuals: 
        return x, count + 1, r[:,:ind+1]
    if not return_residual_norms and not return_residuals: 
        return x, count + 1
    return Q, H
    """






def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
