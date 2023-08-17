import numpy as np
import pytest
import matplotlib.pyplot as plt
from cla_utils.exercises9 import pure_QR
from cla_utils.exercises8 import hessenberg

###part b
A = 1/(sum(np.indices((5,5)))+3)
#print(A)
eigs = np.linalg.eigvals(A)
C,_ = pure_QR(A,maxit=100,tol=1.0e-6,symmetric=True)
#print(C,_)
#print(eigs)
#print(100*abs(eigs-np.diag(C))/abs(eigs)) #percentage difference
#for i in eigs:
        #print(np.linalg.det(A - i*np.eye(5)))

###part c

def partc(A, maxit, tol,shifted=False):
    hessenberg(A)
    m = A.shape[0]
    eigs = []
    belowdiag = np.array([])
    for k in range(m-1,-1,-1):
        C,ar = pure_QR(A,maxit,tol,symmetric=True, shifted=shifted)
        eigs.append(C[k,k])
        A = C[:k,:k]
        if k != 0:
            belowdiag = np.concatenate([belowdiag,ar])
    return np.array(eigs), belowdiag

#plotting sawtooth
"""
rand = np.random.rand(5,5)
randa = rand + rand.T
N = 5
b = np.random.random_integers(-100,100,size=(N,N))
newa = (b + b.T)/2
print(A)
#print(partc(A,maxit=100000,tol=1.0e-05))
_,y = partc(A,maxit=100000,tol=1.0e-05)
x = np.array([i for i in range(len(y))])
plt.figure()
plt.scatter(x,y,s=1)
plt.axhline(y=10e-12, color='r', linestyle='--')
plt.yscale("log")
plt.title("|T_k,k+1| at each iteration of QR")
plt.ylabel("|T_k,k+1|")
plt.xlabel("QR iteration")
plt.show()
"""
#timing modified vs unmodified
"""
import timeit

start = timeit.default_timer()

pure_QR(A,maxit=10000,tol=10e-05,symmetric=True)

stop = timeit.default_timer()

print('Time: ', stop - start)  
start = timeit.default_timer()

pure_QR(A,maxit=10000,tol=10e-05,symmetric=False)

stop = timeit.default_timer()

print('Time: ', stop - start) 

modtimes = []
unmodtimes = []
matsize = [i for i in range(2,50)]
for i in matsize:
    b = np.random.rand(i,i)
    S = (b + b.T)/2
    S0 = 1.0*S
    S1 = 1.0*S
    start = timeit.default_timer()

    pure_QR(S0,maxit=10000,tol=10e-05,symmetric=True)

    stop = timeit.default_timer()

    modtimes.append(stop-start)
    start = timeit.default_timer()

    pure_QR(S1,maxit=10000,tol=10e-05,symmetric=False)

    stop = timeit.default_timer()

    unmodtimes.append(stop-start)
plt.figure()
plt.scatter(matsize,modtimes,label="modified")
plt.scatter(matsize, unmodtimes,label="unmodified")
plt.legend()
plt.title("Timing modified and unmodified QR for matrices of various sizes")
plt.xlabel("Matrix dimension")
plt.ylabel("Time taken to complete QR")
#plt.show()
"""
@pytest.mark.parametrize('m', [2,5,8])
def test_partc(m):
    np.random.seed(323*m)
    b = np.random.rand(m,m)
    S = (b + b.T)/2
    S0 = 1.0*S
    evals,_ = partc(S0, maxit=10000,tol=10e-07)
    for i in evals:
        assert(np.linalg.det(S - i*np.eye(m)) < 10e-4)
###part d
"""
A3 = 1.0*A
A4 = 1.0*A
A5 = 1.0*A
A6 = 1.0*A
print(partc(A3,maxit=100000,tol=1.0e-05))
print(partc(A4,maxit=100000,tol=1.0e-05,shifted=True))
_,y1 = partc(A5,maxit=100000,tol=1.0e-05)
_,y2 = partc(A6,maxit=100000,tol=1.0e-05,shifted=True)
x1 = np.array([i for i in range(len(y1))])
x2 = np.array([i for i in range(len(y2))])
plt.figure()
plt.plot(x1,y1,label="non-shifted")
plt.plot(x2,y2,label="shifted")
plt.legend()
plt.axhline(y=10e-12, color='r', linestyle='--')
plt.yscale("log")

plt.title("|T_k,k+1| at each iteration of QR comparing shifted and non-shifted")
plt.ylabel("|T_k,k+1|")
plt.xlabel("QR iteration")

A = np.diag(np.array([i for i in range(15,0,-1)])) + np.ones((15,15))
print(A)
A3 = 1.0*A
A4 = 1.0*A
A5 = 1.0*A
A6 = 1.0*A
print(partc(A3,maxit=100000,tol=1.0e-05))
print(partc(A4,maxit=100000,tol=1.0e-05,shifted=True))
_,y1 = partc(A5,maxit=100000,tol=1.0e-05)
_,y2 = partc(A6,maxit=100000,tol=1.0e-05,shifted=True)
x1 = np.array([i for i in range(len(y1))])
x2 = np.array([i for i in range(len(y2))])
plt.figure()
plt.plot(x1,y1,label="non-shifted")
plt.plot(x2,y2,label="shifted")
plt.legend()
plt.axhline(y=10e-12, color='r', linestyle='--')
plt.yscale("log")
plt.xlim((0,40))
plt.title("|T_k,k-1| at each iteration of QR comparing shifted and non-shifted")
plt.ylabel("|T_k,k-1|")
plt.xlabel("QR iteration")
plt.show()
"""
@pytest.mark.parametrize('m', [2,5,8])
def test_partcshifted(m):
    np.random.seed(323*m)
    b = np.random.rand(m,m)
    S = (b + b.T)/2
    S0 = 1.0*S
    evals,_ = partc(S0, maxit=10000,tol=10e-07,shifted=True)
    for i in evals:
        assert(np.linalg.det(S - i*np.eye(m)) < 10e-4)

    





