import numpy as np
from cla_utils.exercises5 import operator_2_norm
def eigvals(A):
    tra = A[0,0] + A[1,1]
    det = A[0,0] * A[1,1] - A[1,0] * A[0,1]
    return 0.5*np.array([tra+np.sqrt(tra**2 -4*det), tra-np.sqrt(tra**2 -4*det)])


A1 = np.eye(2)
A2 = np.array([[1 + 10e-14, 0], [0, 1]])
print(eigvals(A1) - np.array([1,1])) #difference in A1 evals
print(eigvals(A2) - np.array([1+10e-14,1])[::-1]) #difference in A2 evals
print(np.linalg.norm(eigvals(A1) - np.array([1+10e-14,1]))/np.linalg.norm(np.array([1+10e-14,1])))
print(np.linalg.norm(A2-A1)/np.linalg.norm(A1))
#[0. 0.]
#[-4.99600361e-14  4.99600361e-14]
#7.065416064076635e-14
#7.065416064076987e-14
