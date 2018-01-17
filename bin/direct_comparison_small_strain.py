## Most of the following is copied from https://github.com/tdegeus/GooseFFT (MIT license)

import numpy as np
import scipy.sparse.linalg as sp
import itertools

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "language_bindings/python"))
import pyMuSpectre as µ


# ----------------------------------- GRID ------------------------------------

ndim   = 3            # number of dimensions
N      = 11 #31  # number of voxels (assumed equal for all directions)
offset = 3 #9
ndof   = ndim**2*N**3 # number of degrees-of-freedom

cell = µ.SystemFactory(µ.get_3d_cube(N),
                       µ.get_3d_cube(1.),
                       µ.Formulation.small_strain)

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta  = lambda i,j: np.float(i==j)            # Dirac delta function
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) # zero initialize
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y,z    in itertools.product(range(N),   repeat=3):
        q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y,z] = -(q[i]*q[j]*q[l]*q[m])/(q.dot(q))**2+\
             (delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l]+\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l])/(2.*q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))

# functions for the projection 'G', and the product 'G : K : eps'
G        = lambda A2   : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_deps   = lambda depsm: ddot42(K4,depsm.reshape(ndim,ndim,N,N,N))
G_K_deps = lambda depsm: G(K_deps(depsm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)

phase  = np.zeros([N,N,N]); phase[-offset:,:offset,-offset:] = 1.
phase  = np.zeros([N,N,N]); phase[0,:,:] = 1.
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones([N,N,N])*(1.-phase)+M1*np.ones([N,N,N])*phase
# K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
# mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]
E2, E1 = 210e9, 70e9
poisson = .33
K2, K1 = (E/(3*(1-2*poisson)) for E in (E2, E1))
m2, m1 = (E/(2*(1+poisson)) for E in (E2, E1))
K =  param(K1, K2)
mu = param(m1, m2)

# stiffness tensor                                            [grid of tensors]
K4     = K*II+2.*mu*(I4s-1./3.*II)

hard = µ.material.MaterialHooke3d.make(cell, "hard",
                                       E2, poisson)
soft = µ.material.MaterialHooke3d.make(cell, "soft",
                                       E1, poisson)

for pixel in cell:
    if ((pixel[0] >= N-offset) and
        (pixel[1] < offset) and
        (pixel[2] >= N-offset)):
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)



# ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain tensor                         [grid of tensors]
sig      = np.zeros([ndim,ndim,N,N,N])
eps      = np.zeros([ndim,ndim,N,N,N])

# set macroscopic loading
DE       = np.zeros([ndim,ndim,N,N,N])
DE[0,1] += 0.01
DE[1,0] += 0.01

µDE = np.zeros([ndim**2, cell.size()])
cell.evaluate_stress_tangent(µDE);
µDE[:,:] = DE[:,:,0,0,0].reshape([-1, 1])

def get_diff_norm(arr_g, arr_mu):
    error = 0
    if arr_g.size == ndim**2*N**3:
        narr_g = arr_g.reshape(ndim,ndim,N,N,N)
        ssiz = ndim
    elif arr_g.size == ndim**4*N**3:
        narr_g = arr_g.reshape(ndim*ndim,ndim*ndim,N,N,N)
        ssiz = ndim**2
    for i, j, l in itertools.product(range(N), repeat=3):
        diff = (narr_g[:,:,i,j,l] -
                arr_mu[:, µ.get_index(µ.get_3d_cube(N), [i,j,l])].reshape([ssiz, ssiz]))
        error += np.linalg.norm(diff)
    return error
def check_equality(arr_g, arr_mu, tol):
        return get_diff_norm(arr_g, arr_mu) < tol

print("Diff norm = {}".format (get_diff_norm(DE, µDE)))

# initial residual: distribute "DE" over grid using "K4"
b        = -G_K_deps(DE)
G_K_deps2 = lambda de: cell.directional_stiffness(de)
b2       = -G_K_deps2(µDE)
err = get_diff_norm(b, b2)
Ghatµ = cell.get_G()
print("Ghat difference = {}".format (get_diff_norm(Ghat4, Ghatµ)))
print("err = {}, b2: min = {}, max = {}".format(err, abs(b2).min(), abs(b2).max()))
print("          b:  min = {}, max = {}".format(     abs(b ).min(), abs(b ).max()))
eps     +=           DE
En       = np.linalg.norm(eps)
iiter    = 0

# iterate as long as the iterative update does not vanish
class accumul(object):
    def __init__(self):
        self.counter = 0
    def __call__(self, dummy):
        self.counter += 1

acc = accumul()
while True:
    depsm,_ = sp.cg(tol=1.e-8,
                    A = sp.LinearOperator(shape=(ndof,ndof),matvec=G_K_deps,dtype='float'),
                    b = b,
                    callback=acc
    )                                     # solve linear system using CG
    eps += depsm.reshape(ndim,ndim,N,N,N) # update DOFs (array -> tens.grid)
    sig  = ddot42(K4,eps)                 # new residual stress
    b     = -G(sig)                       # convert residual stress to residual
    print('%10.2e'%(np.max(depsm)/En))    # print residual to the screen
    if np.linalg.norm(depsm)/En<1.e-5 and iiter>0: break # check convergence
    iiter += 1

print("nb_cg: {0}".format(acc.counter))
