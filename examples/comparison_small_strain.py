#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Most of the following is copied from:
https://github.com/tdegeus/GooseFFT (MIT license)
"""
import sys
import numpy as np
import scipy.sparse.linalg as sp
import itertools

from python_example_imports import muSpectre as µ


# ----------------------------------- GRID ------------------------------------

ndim   = 2            # number of dimensions
N      = 31 #31  # number of voxels (assumed equal for all directions)
offset = 3 #9
ndof   = ndim**2*N**2 # number of degrees-of-freedom
nb_grid_pts = [N, N]
lengths = [1. , 1.]
formulation = µ.Formulation.small_strain

cell = µ.Cell(nb_grid_pts, lengths, formulation)
center = np.array([r // 2 for r in nb_grid_pts])
incl = nb_grid_pts[0] // 5

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ij...          ->ji...  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijkl...,lk...  ->ij...  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijkl...,lkmn...->ijmn...',A4,B4)
dot22  = lambda A2,B2: np.einsum('ij...  ,jk...  ->ik...  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ij...  ,jkmn...->ikmn...',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijkl...,lm...  ->ijkm...',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ij...  ,kl...  ->ijkl...',A2,B2)

shape = tuple((N for _ in range(ndim)))
# identity tensor                                               [single tensor]
i      = np.eye(ndim)
def expand(arr):
    new_shape = (np.prod(arr.shape), np.prod(shape))
    ret_arr = np.zeros(new_shape)
    ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
    return ret_arr.reshape((*arr.shape, *shape))

# identity tensors                                            [grid of tensors]
I     = expand(i)
I4    = expand(np.einsum('il,jk',i,i))
I4rt  = expand(np.einsum('ik,jl',i,i))
I4s   = (I4+I4rt)/2.
II    = dyad22(I,I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta  = lambda i,j: np.float(i==j)            # Dirac delta function
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N]) # zero initialize
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y    in itertools.product(range(N),   repeat=ndim):
        q = np.array([freq[x], freq[y]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y] = -(q[i]*q[j]*q[l]*q[m])/(q.dot(q))**2+\
             (delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l]+\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l])/(2.*q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),shape))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),shape))

# functions for the projection 'G', and the product 'G : K : eps'
G        = lambda A2   : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_deps   = lambda depsm: ddot42(K4,depsm.reshape(ndim,ndim,N,N))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)

E2, E1 = 75e10, 70e9
poisson = .33

hard = µ.material.MaterialLinearElastic1_2d.make(cell, "hard",
                                                 E2, poisson)
soft = µ.material.MaterialLinearElastic1_2d.make(cell, "soft",
                                                 E1, poisson)

#for pixel in cell:
#    if ((pixel[0] >= N-offset) and
#        (pixel[1] < offset)):
#        hard.add_pixel(pixel)
#    else:
#        soft.add_pixel(pixel)
#
phase  = np.zeros(shape); phase[-offset:,:offset,] = 1.
phase  = np.zeros(shape); phase[0,:] = 1.

phase  = np.zeros(shape);

for i, pixel in cell.pixels.enumerate():
    c = N//2
    if np.linalg.norm(center - np.array(pixel), 2) < incl:
        phase[pixel[0],pixel[1]] = 1.
        hard.add_pixel(i)
    else:
        soft.add_pixel(i)
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones(shape)*(1.-phase)+M1*np.ones(shape)*phase
# K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
# mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]
K2, K1 = (E/(3*(1-2*poisson)) for E in (E2, E1))
m2, m1 = (E/(2*(1+poisson)) for E in (E2, E1))
K =  param(K1, K2)
mu = param(m1, m2)

# stiffness tensor                                            [grid of tensors]
K4     = K*II+2.*mu*(I4s-1./3.*II)




# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain tensor                         [grid of tensors]
sig      = np.zeros([ndim,ndim,N,N])
eps      = np.zeros([ndim,ndim,N,N])

# set macroscopic loading
DE       = np.zeros([ndim,ndim,N,N])
DE[0,1] += 0.01
DE[1,0] += 0.01
delEps0 = DE[:ndim, :ndim, 0, 0]

µDE = np.zeros([ndim**2, cell.nb_pixels])
µDE = np.zeros ((ndim, ndim) + (N, N))
cell.evaluate_stress_tangent(µDE);

# initial residual: distribute "DE" over grid using "K4"
b        = -G_K_deps(DE)
G_K_deps2 = lambda de: cell.directional_stiffness(de.reshape(µDE.shape)).ravel()
b2       = -G_K_deps2(µDE).ravel()
print("b2.shape = {}".format(b2.shape))

eps     +=           DE
En       = np.linalg.norm(eps)
iiter    = 0

# iterate as long as the iterative update does not vanish
class accumul(object):
    def __init__(self):
        self.counter = 0
    def __call__(self, x):
        self.counter += 1
        print("at step {}: ||Ax-b||/||b|| = {}".format(
            self.counter,
            np.linalg.norm(G_K_deps(x)-b)/np.linalg.norm(b)))

acc = accumul()
cg_tol = 1e-8
tol = 1e-5
equi_tol = 1e-5
maxiter = 60
solver = µ.solvers.KrylovSolverCGEigen(cell, cg_tol, maxiter,
                                       verbose=µ.Verbosity.Silent)
# solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose=True)
print(delEps0.shape)
try:
    r = µ.solvers.newton_cg(cell, delEps0, solver, tol, equi_tol,
                            verbose=µ.Verbosity.Silent)
except Exception as err:
    print(err)

while True:
    depsm,_ = sp.cg(tol=cg_tol,
                    A = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,
                                          dtype='float'),
                    b = b,
                    callback=acc
    )                                     # solve linear cell using CG
    #depsm2,_ = sp.cg(tol=1.e-8,
    #                 A = sp.LinearOperator(shape=(ndof,ndof),
    #                                       matvec=G_K_deps2,dtype='float'),
    #                 b = b2,
    #                 callback=acc
    #)                                     # solve linear cell using CG
    eps += depsm.reshape(ndim,ndim,N,N) # update DOFs (array -> tens.grid)
    sig  = ddot42(K4,eps)                 # new residual stress
    b     = -G(sig)                       # convert residual stress to residual
    print('%10.2e'%(np.max(depsm)/En))    # print residual to the screen
    print(np.linalg.norm(depsm)/np.linalg.norm(En))
    if np.linalg.norm(depsm)/En<1.e-5 and iiter>0: break # check convergence
    iiter += 1

print("nb_cg: {0}".format(acc.counter))

# prevent visual output during ctest
if len(sys.argv[:]) == 2:
    if sys.argv[1] != 1:
        pass
else:
    import matplotlib.pyplot as plt
    s11 = sig[0,0, :,:]
    s22 = sig[1,1, :,:]
    s21_2 = sig[0,1, :, :]*sig[1,0,:, :]
    vonM1 = np.sqrt(.5*((s11-s22)**2) + s11**2 + s22**2 + 6*s21_2)

    plt.figure()
    img = plt.pcolormesh(vonM1)#eps[0,1,:,:])
    plt.title("goose")
    plt.colorbar(img)

    try:
        print(r.stress.shape)
        arr = r.stress.T.reshape(N, N, ndim, ndim)
        s11 = arr[:,:,0,0]
        s22 = arr[:,:,1,1]
        s21_2 = arr[:,:,0,1]*arr[:,:,1,0]
        vonM2 = np.sqrt(.5*((s11-s22)**2) + s11**2 + s22**2 + 6*s21_2)

        plt.figure()
        plt.title("µSpectre")
        img = plt.pcolormesh(vonM2)#eps[0,1,:,:])
        plt.colorbar(img)
        print("err = {}".format (np.linalg.norm(vonM1-vonM2)))
        print("err2 = {}".format (np.linalg.norm(vonM1-vonM1.T)))
        print("err3 = {}".format (np.linalg.norm(vonM2-vonM2.T)))

        plt.figure()
        plt.title("diff")
        img = plt.pcolormesh(vonM1-vonM2.T)#eps[0,1,:,:])
        plt.colorbar(img)
    except Exception as err:
        print(err)
        plt.show()
