import numpy as np
import muGrid

engine = muGrid.FFTEngine([16, 20])

real_field = engine.real_space_field("real")
fourier_field = engine.fourier_space_field("fourier")

# Initialize real-space field
real_field.s[:] = np.random.randn(1, 1, 16, 20)
original = real_field.s.copy()

# Forward FFT: real -> Fourier
engine.fft(real_field, fourier_field)

# Inverse FFT: Fourier -> real
engine.ifft(fourier_field, real_field)

# Apply normalization for roundtrip
real_field.s[:] *= engine.normalisation

# Verify roundtrip
np.testing.assert_allclose(real_field.s, original, atol=1e-14)
