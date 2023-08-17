import numpy as np
import sys, os
import pytest
sys.path.append(os.path.abspath(os.path.join('..', 'cla_utils')))
import cla_utils
def lagrange_ls(A, b, lam):
    "Solve least squares via Lagrange multiplier"

    Q, R = cla_utils.householder_qr(A)
    Aprime = Q + 1 / lam * A @ R.T
    xprime = cla_utils.householder_solve(Aprime, b)
    return 1 / lam * R.T @ xprime

@pytest.mark.parametrize('m, n', [(3, 23), (40, 130), (87, 90), (3,29), (32, 234), (3, 32), (34,98)])
def test_lagrange_ls(m, n):
    np.random.seed(8473*m + 983*n)
    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)
    lam = np.random.randn(1)[0]

    x = lagrange_ls(A, b, lam)
    x_test = np.random.randn(len(x))
    x_test = x_test / np.linalg.norm(x_test)
    

    #check our residual is smaller than for residual using any other x with norm 1
    assert((np.linalg.norm(A @ x - b))  ** 2 < (np.linalg.norm(A @ x_test - b)) ** 2) 

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)

def find_lambda(A, b):
    
    #testing negative lambda
    top, bottom = -0.001, -10000
    for i in range(100):
        middle = (top + bottom) / 2
        xnorm = np.linalg.norm(lagrange_ls(A, b, middle))
        if xnorm < 1:
            bottom = middle
        if xnorm > 1:
            top = middle
        if xnorm == 1:
            return middle
        if np.linalg.norm(xnorm - 1) < 1.0e-6:
            break
    lneg = middle
    #testing positive lambda
    top, bottom =  10000, 0.001
    for i in range(100):
        middle = (top + bottom) / 2
        xnorm = np.linalg.norm(lagrange_ls(A, b, middle))
        if xnorm < 1:
            bottom = middle
        if xnorm > 1:
            top = middle
        if xnorm == 1:
            return middle
        if np.linalg.norm(xnorm - 1) < 1.0e-6:
            break
    lpos = middle
    p = abs(np.linalg.norm(lagrange_ls(A, b , lpos)) - 1)
    n = abs(np.linalg.norm(lagrange_ls(A, b , lneg)) - 1)
    #print(p, n)
    if p < n:
        return lpos
    else:
        return lneg


@pytest.mark.parametrize('m, n', [(3, 23), (20, 70), (40, 130), (3,29), (3, 32), (23, 24), (34,98)])
def test_find_lambda(m, n):
    np.random.seed(83*m + 9823*n)
    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)
    lam = find_lambda(A, b)
    
    #check our residual is smaller than for residual using any other x with norm 1
    assert(abs(np.linalg.norm(lagrange_ls(A, b , lam)) - 1) < 0.001)


#exploring algoritm for various A
q4data11 = []
q4data12 = []
for i in range(100):
    np.random.seed(83*i + 23*i ** 2)
    m, n = np.random.randint(low=1, high=100, size = 2)
    if m > n:
        m, n = n, m
    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)
    q4data11.append(find_lambda(A, b))
    q4data12.append((m,n))


#writing to file
a_file = open("q4data11.txt", "w")
np.savetxt(a_file, q4data11)
a_file.close()
a_file = open("q4data12.txt", "w")
np.savetxt(a_file, q4data12)
a_file.close()