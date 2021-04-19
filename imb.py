'''
Immersed Boundary methods

Dependencies:
    numpy 1.19.4

Programmed by Manuel Giménez de Castro. Advisor: Prof. Dr. Alexandre Megiorin Roma.

General Description:
    Recieves velocity field and membrane position in form of an ndarray. Calculates the force in the eulerian grid and velocity in 
    the lagrangian points. 

Physical parameters:
    T: tension of the immersed boundary: float
    X: array with the x position of the langrangian points: ndarray 
    Y: array with the y position of the langrangian points: ndarray
    U: velocity field in the x direction: ndarray
    V: velocity field in the y direction: ndarray
Computational parameters:
    M: number of points of the discrete boundary: int
    N: number of points in a single direction: int
    dh: step of the eulerian grid: float
    ds: step of the lagrangian points: float
Notes: 

'''

import numpy as np

def phi(x):
	if np.abs(x) < 2:
		return (1+np.cos(np.pi*x/2))/4
	else:
		return 0

def delta_h(x,y,dh):
	return phi(x/dh)*phi(x/dh)/dh**2

'''
Peskin93 Dirac's Delta in its matricial form of the phi function
recieves:
    x: matrix with values: NDARRAY
returns:
    res: matrix with answer: NDARRAY
'''
def phi2(x):
	a=(np.abs(x) < 2)
	res = np.zeros(x.shape)
	res[a] = (1+np.cos(np.pi*x[a]/2))/4
	res[~a] = 0
	return res

'''
2d delta function with the matricial form
recieves:
    x: matrix in x direction: NDARRAY
	y: matrix in y direction: NDARRAY
returns:
    matrix with answer: NDARRAY
'''
def delta_h2(x,y,dh):
    return phi2(x/dh)*phi2(y/dh)/dh**2

'''
Interpolate velocity using the Dirac's Delta methodology.
recieves:
    U: Velocity in x direction: NDARRAY
	V: Velocity in y direction: NDARRAY
	X: matrix with x position of the boundary: NDARRAY (1-D) 
	Y: matrix with y position of the boundary: NDARRAY (1-D) 
	dt: time step: FLOAT
	dh: spacial step: FLOAT
returns:
    XRES: x position of the boundary: NDARRAY (1-D)
    YRES: y position of the boundary: NDARRAY (1-D)
'''
def interpolate_velocity(U,V,X,Y,dt,dh):
    if not np.allclose(X.shape,Y.shape):
        raise ValueError("[ERROR] Immersed boundary shapes don't match.")
    if not np.allclose(U.shape,V.shape):
        raise ValueError("[ERROR] Matrices sizes don't match.")

    M = X.shape[0]
    #new position 
    XRES = X.copy()
    YRES = Y.copy()

    for m in range(M):
        i1 = int(X[m]/dh)-1 #this defines the 4x4 box that can differ from 0  
        j1 = int(Y[m]/dh)-1
        i2 = i1+3
        j2 = j1+3
        if i1 < 0 :
            i1 = 0
        if j1 < 0:
            j1 = 0
        if i2 > U.shape[0]-1:
            i2 = U.shape[0]-1
        if j2 > U.shape[0]-1:
            j2 = U.shape[0]-1

        #integrate via midpoint rule only the non zero entries of the equation
        grid = np.meshgrid(np.array(list(range(i1,i2+1)))*dh,np.array(list(range(j1,j2+1)))*dh) #only supports if the spacial interval is [0,A], A being wathever
        delta_matrix = delta_h2(grid[0]-X[m],grid[1]-Y[m],dh)
        XRES[m] += dt*dh**2*np.sum(U[j1:j2+1,i1:i2+1]*delta_matrix) #displacement of the first coordinate integrated via midpoint rule
        YRES[m] += dt*dh**2*np.sum(V[j1:j2+1,i1:i2+1]*delta_matrix) #displacement of the second coordinate of the interface

    return XRES,YRES

'''
Receives the current velocity field and immersed boundary points. Calculates the force and spreads it using the dirac's delta in one 
of its discrete forms.

WATCH OUT TO NOT PUT THE IMMERSED BOUNDARY ON THE RIGHT AND BOTTOM OF THE PERIODIC BOX! THE MATRIX WILL BE THE SAME SIZE AS U AND V. 

recieves:
    U: Velocity in x direction: NDARRAY
	V: Velocity in y direction: NDARRAY
	X: matrix with x position of the boundary: NDARRAY (1-D) 
	Y: matrix with y position of the boundary: NDARRAY (1-D) 
	T: non negative constant of the tension: FLOAT
	dh: spacial step: FLOAT
	ds: boundary step: FLOAT
	debug: switches to a constant force field: BOOL
returns:
    XRES: x position of the boundary: NDARRAY (1-D)
    YRES: y position of the boundary: NDARRAY (1-D)
'''
def spread_force(U,V,X,Y,T,dh,ds,debug = False):
    if not np.allclose(X.shape, Y.shape):
        raise ValueError("[ERROR] Immersed boundary shapes don't match.")
    if not np.allclose(U.shape, V.shape):
        raise ValueError("[ERROR] Matrices sizes don't match.")
    
    c01 = T/ds**2
    M = X.shape[0]
    FX = np.zeros(U.shape) #where the calculated force vector field is stored
    FY = np.zeros(V.shape)

    if debug:
        FBX = np.full(M,0.1)
        FBY = np.full(M,0.2)
    else:
        FBX = np.zeros(M)
        FBY = np.zeros(M)
        for i in range(M):
            FBX[i] = c01*(X[(i+1)%M]-2*X[i]+X[i-1]) #force across the boundary
            FBY[i] = c01*(Y[(i+1)%M]-2*Y[i]+Y[i-1]) #second order discretization of the curvature of the boundary

    for m in range(M):
        i1 = int(X[m]/dh)-1 #this defines the 4x4 box that can differ from 0  
        j1 = int(Y[m]/dh)-1
        i2 = i1+3
        j2 = j1+3
        if i1 < 0 :
            i1 = 0
        if j1 < 0:
            j1 = 0
        if i2 > U.shape[0]-1:
            i2 = U.shape[0]-1
        if j2 > U.shape[0]-1:
            j2 = U.shape[0]-1

        #only supports if the spacial interval begins at 0! 
        grid = np.meshgrid(np.array(list(range(i1,i2+1)))*dh,np.array(list(range(j1,j2+1)))*dh) 
        delta_matrix = delta_h2(grid[0]-X[m],grid[1]-Y[m],dh)

        #integrate via midpoint rule only the non zero entries of the Dirac delta matrix 
        FX[j1:j2+1,i1:i2+1] += ds*FBX[m]*delta_matrix
        FY[j1:j2+1,i1:i2+1] += ds*FBY[m]*delta_matrix
    return FX,FY
