import numpy as np
import muGrid
from muGrid import fft_real_space_field, fft_fourier_space_field

# Grid parameters
nx, ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
dx, dy = Lx / nx, Ly / ny

engine = muGrid.FFTEngine([nx, ny])

# Create fields
f = fft_real_space_field(engine, "f")
f_hat = fft_fourier_space_field(engine, "f_hat")
grad_x = fft_real_space_field(engine, "grad_x")
grad_y = fft_real_space_field(engine, "grad_y")
grad_hat = fft_fourier_space_field(engine, "grad_hat")

# Initialize with a smooth function
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
f.s[0, 0, :, :] = np.sin(X) * np.cos(Y)

# Forward FFT
engine.fft(f._cpp, f_hat._cpp)

# Compute wavevectors
kx = 2 * np.pi * np.array(muGrid.rfft_freqind(nx)) / Lx
ky = 2 * np.pi * np.array(muGrid.fft_freqind(ny)) / Ly
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# Derivative in x: multiply by i*kx
grad_hat.s[0, 0, :, :] = 1j * KX * f_hat.s[0, 0, :, :]
engine.ifft(grad_hat._cpp, grad_x._cpp)
grad_x.s[:] *= engine.normalisation

# Derivative in y: multiply by i*ky
grad_hat.s[0, 0, :, :] = 1j * KY * f_hat.s[0, 0, :, :]
engine.ifft(grad_hat._cpp, grad_y._cpp)
grad_y.s[:] *= engine.normalisation

# Verify: d/dx[sin(x)cos(y)] = cos(x)cos(y)
expected_grad_x = np.cos(X) * np.cos(Y)
np.testing.assert_allclose(grad_x.s[0, 0], expected_grad_x, atol=1e-10)
