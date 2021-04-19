#imb

The Immersed Boundary Method, imb, is an algorithm to solve Navier-Stokes equations with the interaction of _fully immersed elastic 
structures_. 

## Implementation

The whole algorithm is programmed in **Python 3.6.9** using the following python libraries and versions:

- [NumPy 1.19.4](https://NumPy.org/),
- [pyFFTW 0.12.0](https://github.com/pyFFTW/pyFFTW).

Other versions of Python and auxiliary libraries might work, but haven't been tested.

## Installation and Usage

Make sure you have the dependencies:

```
pip3 install pyfftw && pip3 install numpy
```

Download the files in put them in the same directory as your Python3 script. Import nssolver and imb libraries. Note that you can use 
separarely just the Navier-Stokes solver if you wish.

```
import nssolver,imb
```

Each function and its usage is explained in comments on each file.

## History

This method was proposed by professor [Charles Peskin](https://www.math.nyu.edu/faculty/peskin/) et al in a 
[series of papers](https://www.math.nyu.edu/faculty/peskin/ib_lecture_notes/index.html). The method uses the well known Navier-Stokes 
equations for a viscid fluid with incompressible flow coupled with specific equations that consider the motion and force of the 
immersed structure. 

This was the code used in the author's simulations presented in his final work to obtain the title of Bachelor in Applied Mathematics 
by the University of São Paulo with professor Alexandre Megiorin Roma ([Curriculum Vitae](http://lattes.cnpq.br/4149882391730362) in 
portuguese) as the advisor. If desired, the whole work with the deduction of the equations solved plus the numerical scheme is 
available in the author's [homepage](http://www.manuel.gcastro.net/texts/).

Is worth noting that this is just a first implementation, with multiple obvious setbacks. A far better implementation to use in real 
case scenarios is professor [Boyce Griffith](https://cims.nyu.edu/~griffith/)'s et al [IBAMR](https://github.com/IBAMR/IBAMR). The 
sole intention of the repository is to put the source code for other students that might be interested in a simple implementation for 
learning purposes.
