'''
File that generates a movie of the final simulation. We start with a zero field velocity and a compressed tube. As the tube returns 
to its resting position it moves the fluid around it.

Dependencies:
	nssolver
	imb
	numpy 1.19.4
	matplotlib.animation
	pyplot
	time 

Programmed by Manuel Giménez de Castro. Advisor: Prof. Dr. Alexandre Megiorin Roma.

General Description:
	This file calculates the simulation using nssolver and imb methods.

Physical parameters:
	r: mass density
	mu: dynamical viscosity
	T: Non negative constant that defines the tension.

Computational parameters:
	N: number of grid points direction x AND y
	N_t: number of time steps
	M: number of points of the lagrangian discretization
Notes: 
	This program uses the main memory. Be carefull with the no_saved setting and precision of the integration.
'''

import nssolver
import imb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

'''
Places points in an ellipse
recieves:
    t: parameter of the position of ellipse: FLOAT
    a: x axis lenght of the ellipse: FLOAT
    b: y axis lenght of the ellipse: FLOAT
    center: position of the ellipse center: NDARRAY
returns:
    [x,y]: position of the ellipse: NDARRAY
'''
def set_elipse(t,a,b,center):
	x = center[0]+a*np.cos(t)
	y = center[1]+b*np.sin(t)
	return np.array([x,y])

'''
Copyies the left and top of the matrices to the right and bottom. This is because of the periodic condition.
recieves:
    A: (N,N) matrix to be enlarged: NDARRAY
returns:
    B: (N+1,N+1) matrix with mirrored left and top: NDARRAY 
'''
def enlarge(A):
	#creates a N+1 x N+1 matrix
	N = A.shape[1]
	B = np.zeros(np.array(A.shape)+1)
	#copy the whole A matrix.
	B[:N,:N] = A
	B[-1,:-1] = A[0,:] #copy the top of the matrix to the bottom
	B[:-1,-1] = A[:,0] #copy the left of the matrix to the left
	B[-1,-1] = A[0,0]
	return B

if __name__ == "__main__":
	'''
    Physical constants
	'''
	r = 1 #(g/cm^2)
	mu = 0.01 #g/(cm*s)
	T = 1 #(g*cm)/s^2	

	verbose = True
	#verbose = False
	#adjust_mean = True
	adjust_mean = False
	save_file = True
	#save_file = False
	flag = False
	no_saved = 8 #number of saved instants for the plot.
	date = time.localtime()


	#computational parameters
	N = 512
	N_t = N
	M = 512
	dh = 1/N
	dt = 1/(N_t-1)
	
	#sets the position of the immersed boundary
	vectorm = np.linspace(0,2*np.pi,M,endpoint=False)
	boundary = set_elipse(vectorm,1/8,1/16,[1/2,1/2])
	ds = 2*np.pi/M
	X = boundary[0]
	Y = boundary[1]

	#every position that will be saved to create the animation	
	URES = np.zeros((no_saved,N,N))
	VRES = np.zeros((no_saved,N,N))	
	XRES = np.zeros((no_saved,M))
	YRES = np.zeros((no_saved,M))	

	#copyies the first position
	XRES[0] = X.copy()
	YRES[0] = Y.copy()

	#saves the last position
	ULAST = URES[0].copy()
	VLAST = VRES[0].copy()
	XLAST = X.copy()
	YLAST = Y.copy()
	
	if verbose:
		print("Plotting a 2 figures with 8 subplots each!")
		print(f"Simulation with {N} spacial points for each direction and {N_t} temporal points.")
		print(f"dh: {dh:.4e}")
		print(f"dt: {dt:.4e}")

	#used to track the maximum velocity
	maxnorm = nssolver.maxnrm(URES[0],VRES[0])
	if maxnorm >= 2*mu/(r*dh) and verbose:
		print(f"\n[WARNING] Estability condition violated at instant 0\n maxnorm = {maxnorm:.4e} >= {2*mu/(r*dh):.4e}.")
		flag = True

	#solves the system
	for t in range(1,N_t):
		FX, FY = imb.spread_force(ULAST,VLAST,XLAST,YLAST,T,dh,ds)
		ULAST,VLAST,PRES = nssolver.nssolve(ULAST,VLAST,FX,FY,dt,dh,mu,r)
		XLAST,YLAST = imb.interpolate_velocity(ULAST,VLAST,XLAST,YLAST,dh,dh)
		if t%(N_t//no_saved) == 0:
			URES[t//(N_t//no_saved)] = ULAST.copy()
			VRES[t//(N_t//no_saved)] = VLAST.copy()
			XRES[t//(N_t//no_saved)] = XLAST.copy()
			YRES[t//(N_t//no_saved)] = YLAST.copy()
		if verbose:
			print(f"Instant {t*dt:.4e} of {(N_t-1)*dt:.4e}", end="\r")
			nssolver.div(ULAST,VLAST,dh) #checks if the divergent is zero
	
	#grid
	vectorx = np.linspace(0,1,N+1)
	grid = np.meshgrid(vectorx,vectorx)
	
	fig, axrr = plt.subplots(4,2,figsize=(15,15))

	for i in range(4):
		for j in range(2):
			axrr[i,j].set_aspect('equal')
			im = axrr[i,j].contourf(grid[0],grid[1],enlarge(np.sqrt(URES[i*2+j]**2+VRES[i*2+j]**2)))
			axrr[i,j].plot(XRES[i*2+j],YRES[i*2+j],'k')
			axrr[i,j].set_title(f"{dt*(N_t//no_saved)*(i*2+j):.4e}")				

	fig.tight_layout()
	#fig.suptitle(f"Simulation of the tube with $T_0 = {T}$, $\\mu = {mu}$, and $\\rho = {r}$.")

	if save_file:
		file_name = f"simulation_contourf_{date[3]}_{date[4]}_{date[5]}.png"
		plt.savefig(file_name, dpi=300)

	plt.clf()	
	fig, axrr = plt.subplots(4,2,figsize=(15,15))

	for i in range(4):
		for j in range(2):
			axrr[i,j].set_aspect('equal')
			im = axrr[i,j].quiver(grid[0],grid[1],enlarge(URES[i*2+j]),enlarge(VRES[i*2+j]))
			axrr[i,j].plot(XRES[i*2+j],YRES[i*2+j],'k')
			axrr[i,j].set_title(f"{dt*(N_t//no_saved)*(i*2+j):.4e}")				
	
	fig.tight_layout()
	#fig.suptitle(f"Simulation of the tube with $T_0 = {T}$, $\\mu = {mu}$, and $\\rho = {r}$.")

	if save_file:
		file_name = f"simulation_quiver_{date[3]}_{date[4]}_{date[5]}.png"
		plt.savefig(file_name, dpi=300)
