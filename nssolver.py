'''
Navier-Stokes incompressible, viscid and constant density solver using Chorin's projection method.

Dependencies:
    numpy 1.19.4
    pyfftw 0.12.0
    ptdsys 
    
Programmed by Manuel Giménez de Castro. Advisor: Prof. Dr. Alexandre Megiorin Roma.

General Description:
    Recieves velocity field and force field and compute the next time step using projection method.

Physical parameters:
    U: velocity field in the direction x: NDARRAY
    V: velocity field in the direction y: NDARRAY
    FX: force field in the direction x: NDARRAY
    FY: force field in the direction y: NDARRAY
    r: mass density: float (*)
    mu: dynamical viscosity: float (*)

Computational parameters:
    dt: time step: float
    N: number of of grid points direction x AND y, this is taken from the matrix shape: int (**)
    dh: spacial step in direction x AND y: float (**)

Notes: 
    (*) These are assumed constant.
    (**) This implementation ONLY supports if LX==LY and NX==NY. 
    This implementation was described in Peskin REFREFREF.
'''

import numpy as np
import pyfftw
import ptdsys

'''
Tests whether incompressibity condition holds at every point of the grid, including the edges of the periodic box, except 
computational error.
recieves:
    U: velocity field in the direction x: NDARRAY
    V: velocity field in the direction y: NDARRAY
    dh: spacial step in direction x and y: float 
returns: 
    None
'''
def div(U,V,dh):
    newdiv = 0
    maxdiv = 0
    for i in range(0,U.shape[0]):
        for j in range(0,U.shape[1]):
            #evaluates divergence in the edge of the periodic box
            if i == 0:
                if j == 0:
                    newdiv = U[j,1]-U[j,-1]+V[j+1,0]-V[j-1,0]
                elif j == U.shape[1]-1:
                    newdiv = U[j,1]-U[j,-1]+V[0,0]-V[j-1,0]
                else:
                    newdiv = U[j,1]-U[j,-1]+V[j+1,0]-V[j-1,0]
            elif i == U.shape[0]-1:
                if j == 0:
                    newdiv = U[0,0]-U[0,-2]+V[1,-1]-V[-1,-1]
                elif j == U.shape[1]-1:
                    newdiv = U[-1,0]-U[-1,-2]+V[0,-1]-V[-2,-1]
                else:
                    newdiv = U[j,0]-U[j,-2]+V[j+1,-1]-V[j-1,-1]
            else:
                if j == 0:
                    newdiv = U[0,i+1]-U[0,i-1]+V[1,i]-V[-1,i]
                elif j == U.shape[1]-1:
                    newdiv = U[-1,i+1]-U[-1,i-1]+V[0,i]-V[-2,i]
                else:
                    newdiv = U[j,i+1]-U[j,i-1]+V[j+1,i]-V[j-1,i]
            if np.abs(newdiv) > np.abs(maxdiv): maxdiv = newdiv
    
    maxdiv /= (2*dh)
   
    #checks if it is numerically close to 0
    if not np.isclose(maxdiv,0):
        print(f"\n[WARNING] Velocity divergent, {maxdiv:.2e}, not close to zero.")

'''
Calculates the maximum norm of the velocity field.
recieves:
    U: velocity field in the direction x: NDARRAY
    V: velocity field in the direction y: NDARRAY
returns: 
    maximum norm: float 
'''
def maxnrm(U,V):
    return max(abs(U).max(),abs(V).max())

'''
From the velocity field in time step n, calculates the next one.
recieves:
    U: velocity field in the direction x and instant n: NDARRAY
    V: velocity field in the direction y and instant n: NDARRAY
    FX: force field in the direction x and instant n: NDARRAY
    FY: force field in the direction y and instant n: NDARRAY
    dt: time step: float
    dh: spacial step in direction x AND y: float (**)
    mu: dynamical viscosity: float (*)
    r: mass density: float (*)
returns:
    URES: velocity field direction x in n+1: NDARRAY
    VRES: velocity field direction y in n+1: NDARRAY
    PRES: pressure field in n+1: NDARRAY
'''

def nssolve(U,V,FX,FY,dt,dh,mu,r):

    #checks if every matrix size matches. Every matrix HAS to be square and SAME size
    if not U.shape == V.shape == FX.shape == FY.shape:
        raise ValueError("[ERROR] Matrices sizes don't match.")

    #number of GRID points on each direction. ONLY SUPPORTS IF NX == NY
    if U.shape[0] != U.shape[1] or V.shape[0] != V.shape[1]:
        raise ValueError("[ERROR] U or V are not square matrices.")

    #number of discretization points
    N = U.shape[0]
    #constants used throughout the code
    c01 = mu*dt/(r*dh**2)
    c02 = 0.5*dt/dh
    c03 = dt/r
    c04 = 2*np.pi/N
    c06 = -(r*dh/dt)*1j

    ########################################
    # step 0: force influence is computed. #
    ########################################
    
    #auxiliary matrices. Used as intermediary information holders
    UA = U + c03*FX #matrix X direction
    VA = V + c03*FY #matrix Y direction
    
    ################################################
    # step 1: viscosity in direction x is computed #
    ################################################

    #U[i,j] \approx u(j * \Delta x,i * \Delta y)
    for j in range(N):
        A = c01+c02*U[j,:] 
        B = np.array(N*[1+2*c01]) 
        C = c01-c02*U[j,:] 

        UA[j,:] = ptdsys.ptdsys(A,B,C,UA[j,:])
        VA[j,:] = ptdsys.ptdsys(A,B,C,VA[j,:])       

    ################################################
    # step 2: viscosity in direction y is computed #
    ################################################

    for i in range(N):
        A = c01+c02*V[:,i]
        B = np.array(N*[1+2*c01]) 
        C = c01-c02*V[:,i] 

        UA[:,i] = ptdsys.ptdsys(A,B,C,UA[:,i])
        VA[:,i] = ptdsys.ptdsys(A,B,C,VA[:,i])

    #############################################################################
    # step 3: velocity vector is projected onto a incompressible velocity field #
    #############################################################################
    
    pyfftw.interfaces.cache.enable()
    #fft matrices
    UHAT = pyfftw.interfaces.numpy_fft.fft2(UA)  
    VHAT = pyfftw.interfaces.numpy_fft.fft2(VA)    
    PHAT = np.zeros(UHAT.shape,dtype='complex')

    for i in range(N):
        for j in range(N):
            c05 = np.sin(c04*j)*UHAT[i,j] + np.sin(c04*i)*VHAT[i,j]
            if not np.isclose((np.sin(c04*i))**2 + (np.sin(c04*j))**2,0):
                c05 /= (np.sin(c04*i))**2 + (np.sin(c04*j))**2
            #populating PHAT, UHAT, and VHAT matrices
            PHAT[i,j] = c05*c06
            UHAT[i,j] -= c05*np.sin(c04*j)
            VHAT[i,j] -= c05*np.sin(c04*i)

    URES = np.real(pyfftw.interfaces.numpy_fft.ifft2(UHAT))
    VRES = np.real(pyfftw.interfaces.numpy_fft.ifft2(VHAT)) 
    PRES = np.real(pyfftw.interfaces.numpy_fft.ifft2(PHAT))

    return URES,VRES,PRES
