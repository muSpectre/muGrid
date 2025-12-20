/**
 * @file   fft/fft_backend_factory.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   20 Dec 2024
 *
 * @brief  Factory functions for FFT backends
 *
 * Copyright © 2024 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "fft_1d_backend.hh"
#include "pocketfft_backend.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include "cufft_backend.hh"
#elif defined(MUGRID_ENABLE_HIP)
#include "hipfft_backend.hh"
#endif

namespace muGrid {

std::unique_ptr<FFT1DBackend> get_host_fft_backend() {
  return std::make_unique<PocketFFTBackend>();
}

std::unique_ptr<FFT1DBackend> get_device_fft_backend() {
#if defined(MUGRID_ENABLE_CUDA)
  return std::make_unique<cuFFTBackend>();
#elif defined(MUGRID_ENABLE_HIP)
  return std::make_unique<hipFFTBackend>();
#else
  return nullptr;
#endif
}

}  // namespace muGrid
