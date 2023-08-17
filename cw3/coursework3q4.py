import numpy as np
from numpy.linalg.linalg import solve
import pytest
from cla_utils import GMRES
from cla_utils import solve_R, solve_L, LUP_inplace
from cla_utils.exercises6 import LU_inplace
from cla_utils.exercises7 import solve_LUP
import scipy



###part a
def serialise(u):
    """
    Also works fine for the d_x and d_y arrays
    """
    v = np.ndarray.flatten(u,order="F")
    return v
print(serialise(np.array([[1,2],[3,4],[5,6]])).shape)
def invserialv(v):
    dim = int(np.sqrt(len(v)))
    u = np.reshape(v,(dim,dim)).T
    return u
print(invserialv(serialise(np.array([[1,2,4],[5,6,7],[7,8,9]]))).shape)
def invseriald(d, x, y):
    dim = int((-1 + int(np.sqrt(1+4*len(d))))/2)
    if x == True:
        return np.reshape(d,(dim,dim+1)).T
    if y == True:
        return np.reshape(d,(dim+1,dim)).T
print(invseriald(np.array([1,3,5, 2, 4, 6]),x=True,y=False))

@pytest.mark.parametrize('m, n', [(4, 4), (3, 13), (21, 8)])
def test_serialise(m,n):
    np.random.seed(323*m + n)
    u = np.random.rand(m,n)
    v = serialise(u)
    assert v.shape[0] == m*n
    for i in range(n):
        assert np.array_equal(u[:,i],v[m*i:m*(i+1)])

@pytest.mark.parametrize('m, n', [(4, 4), (13, 13), (21, 21)])
def test_invserialv(m,n):
    np.random.seed(323*m + n)
    u = np.random.rand(m,n)
    v = serialise(u)
    assert np.array_equal(u,invserialv(v))

@pytest.mark.parametrize('m, n', [(4, 3), (13, 12), (21, 20)])
def test_invserialdx(m,n):
    np.random.seed(323*m + n)
    a = np.random.rand(m,n)
    dx = serialise(a)
    assert np.array_equal(a,invseriald(dx,x=True,y=False))

@pytest.mark.parametrize('m, n', [(2, 3), (11, 12), (19, 20)])
def test_invserialdy(m,n):
    np.random.seed(323*m + n)
    a = np.random.rand(m,n)
    dy = serialise(a)
    assert np.array_equal(a,invseriald(dy,x=False,y=True))

###part b
def H_apply(v,lam=1,mu=1):
    u = invserialv(v)
    u0 = 4.0*u
    u0[1:,:] = u0[1:,:] - u[:-1,:]
    u0[:-1,:] = u0[:-1,:] - u[1:,:]
    u0[:,1:] = u0[:,1:] - u[:,:-1]
    u0[:,:-1] = u0[:,:-1] - u[:,1:]
    u0 = lam*u0
    u0 = u0 + mu*u
    return serialise(u0)
u = np.zeros((4,4))
u[1:3,1:3] = np.array([[1,2],[3,4]])
print(u)
print(H_apply(np.array(serialise(u)),lam=1,mu=1))

###part c
@pytest.mark.parametrize('m', [4, 36, 81])
def test_GMRESmatfunc(m):
    A = None
    b = np.random.randn(m)

    x, _ = GMRES(A, b, maxit=1000, tol=1.0e-3,matfunc=H_apply)
    assert(np.linalg.norm(H_apply(x) - b) < 1.0e-3)

###part d
def LU_banded_inplace(A):
    m = A.shape[0]
    L = np.eye(m)
    p = np.where(A[0,:] == 0)[0][0]
    q = np.where(A[:,0] == 0)[0][0]
    for k in range(m-1):
        for j in range(k+1, min(k+q,m)):
            L[j, k] = A[j, k] / A[k, k]
            n = min(k+p, m)
            A[j, k:n] = A[j, k:n] - L[j, k] * A[k, k:n]
    
    return L, A

def bandedsolver(A,b):
    L,U = LU_banded_inplace(A)
    b = np.reshape(b,(len(b),1))
    y = solve_L(L,b)
    x = solve_R(U,y)
    return x

def M_solve(x,lam=1,mu=1):
    x = invserialv(x)
    m = len(x)
    T = np.diag(2*np.ones([m])*lam+mu*np.ones([m])) + np.diag(-1*np.ones([m-1])*lam,-1) + np.diag(-1*np.ones([m-1])*lam,1)
    for k in range(m):
        T0 = 1.0*T
        x0 = bandedsolver(T0,x[:,k])
        x[:,k] = x0
        
    x = mu*x
    for k in range(m):
        T1 = 1.0*T
        x0 = bandedsolver(T1,x[k,:])
        x[k,:] = x0
    return serialise(x)

    


###part e
@pytest.mark.parametrize('m', [16,36, 81])
def test_GMRESpreconfunc(m):
    A = None
    b = np.random.randn(m)

    x, _ = GMRES(A, b, maxit=10000, tol=1.0e-7,matfunc=H_apply,precondfunc=M_solve)
    assert(np.linalg.norm(H_apply(x) - b) < 1.0e3)

###part f


###part g
#creating an bright empty square silhoutte image
image = np.zeros((10,10))
image[2:-2,2:-2] = np.ones((6,6))*0.9
image[3:-3,3:-3] = np.zeros((4,4))
print(image)
#creating noise
image = image + np.random.rand(10,10)*10e-5
print(image)
