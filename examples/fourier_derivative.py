import numpy as np

import muGrid

# Grid parameters
nx, ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi

engine = muGrid.FFTEngine([nx, ny])

# Create fields
f = engine.real_space_field("f")
f_hat = engine.fourier_space_field("f_hat")
grad_x = engine.real_space_field("grad_x")
grad_y = engine.real_space_field("grad_y")
grad_hat = engine.fourier_space_field("grad_hat")

# Get real-space coordinates from engine
X, Y = engine.coords
X = X * Lx  # Scale from [0, 1) to [0, Lx)
Y = Y * Ly

# Initialize with a smooth function: f = sin(x) * cos(y)
# Note: scalar fields have .s shape (sub_pts, spatial) = (1, nx, ny)
f.s[0, :, :] = np.sin(X) * np.cos(Y)

# Forward FFT
engine.fft(f, f_hat)

# Get wavevectors from engine (integer indices, then scale)
# fftfreq returns normalized frequencies in [-0.5, 0.5)
# Multiply by 2*pi*n/L to get physical wavevectors
QX, QY = engine.fftfreq
KX = 2 * np.pi * QX * nx / Lx  # Physical wavevector
KY = 2 * np.pi * QY * ny / Ly

# Derivative in x: multiply by i*kx
grad_hat.s[0, :, :] = 1j * KX * f_hat.s[0, :, :]
engine.ifft(grad_hat, grad_x)
grad_x.s[:] *= engine.normalisation

# Derivative in y: multiply by i*ky
grad_hat.s[0, :, :] = 1j * KY * f_hat.s[0, :, :]
engine.ifft(grad_hat, grad_y)
grad_y.s[:] *= engine.normalisation

# Verify: d/dx[sin(x)cos(y)] = cos(x)cos(y)
expected_grad_x = np.cos(X) * np.cos(Y)
np.testing.assert_allclose(grad_x.s[0], expected_grad_x, atol=1e-10)

print("Fourier derivative test passed!")
