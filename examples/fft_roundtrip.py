import numpy as np
import muGrid
from muGrid import fft_real_space_field, fft_fourier_space_field

engine = muGrid.FFTEngine([16, 20])

real_field = fft_real_space_field(engine, "real")
fourier_field = fft_fourier_space_field(engine, "fourier")

# Initialize real-space field
real_field.s[:] = np.random.randn(1, 1, 16, 20)
original = real_field.s.copy()

# Forward FFT: real -> Fourier
engine.fft(real_field._cpp, fourier_field._cpp)

# Inverse FFT: Fourier -> real
engine.ifft(fourier_field._cpp, real_field._cpp)

# Apply normalization for roundtrip
real_field.s[:] *= engine.normalisation

# Verify roundtrip
np.testing.assert_allclose(real_field.s, original, atol=1e-14)
