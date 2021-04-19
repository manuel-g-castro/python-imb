'''
File with a tridiagonal periodic solver of linear systems. 

Dependencies:
    numpy 1.19.4
    
Programmed by Manuel Giménez de Castro. Advisor: Prof Dr. Alexandre Megiorin Roma.

General Description:
    Solves Mx = D, where M is a tridiagonal periodic matrix.

Computational parameters:
    A: lower diagonal: NDARRAY
    B: diagonal: NDARRAY
    C: higher diagonal: NDARRAY
    D: right side of the equations: NDARRAY
Notes: 
    This is based on Alexandre Roma's ptdsys.f implemented in FORTRAN.
'''
import numpy as np

def ptdsys(A,B,C,D):
    #checks if every shape is the same
    if not (A.shape == B.shape == C.shape == D.shape):
        raise ValueError("[ERROR] A, B, C or D shapes don't match.")
    
    N = A.shape[0]
    B0 = B.copy()
    C0 = C.copy()
    XA = np.zeros(N+1, dtype='float64')
    X = np.zeros(N+1, dtype='float64') # vector with answer

    XA[0] = XA[N] = 1
    C0[0] = 0

    for j in range(1,N):
        B0[j] -= A[j]*C0[j-1]
        C0[j] /= B0[j]
        X[j] = (D[j]+A[j]*X[j-1])/B0[j]
        XA[j] = (A[j]*XA[j-1])/B0[j]

    for j in range(N-1, 0,-1):
        X[j] += C0[j]*X[j+1]
        XA[j] += C0[j]*XA[j+1]

    L=(A[0]*X[N-1]+C[0]*X[1]+D[0])/(B[0]-A[0]*XA[N-1]-C[0]*XA[1])

    X[0] = L

    for j in range(1,N):
        X[j] += L*XA[j]

    return X[:-1] 
