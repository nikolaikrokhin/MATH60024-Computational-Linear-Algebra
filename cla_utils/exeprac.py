from cla_utils.exercises3 import householder_qr
from exercises7 import solve_LUP
from exercises6 import LU_inplace
from exercises2 import GS_modified_R
from exercises3 import householder_qr
import numpy as np
x = np.random.rand(10,10)
y = np.random.rand(10,10) 
A = x + 1j * y
r = np.random.randn(20)
t = np.random.randn(20)
b = r + 1j * t
A0 = 1.0*A
A1 = 1.0*A

Q, R = householder_qr(A0)


print(Q @ R - A0)


