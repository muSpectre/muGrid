/**
 * @file   fft/fft_utils.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Utility functions for FFT operations
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

#ifndef SRC_LIBMUGRID_FFT_FFT_UTILS_HH_
#define SRC_LIBMUGRID_FFT_FFT_UTILS_HH_

#include "core/types.hh"
#include "core/coordinates.hh"

#include <vector>

namespace muGrid {

/**
 * Compute the frequency bin index for a single position in a full c2c FFT.
 *
 * Maps position i (0 <= i < n) to frequency index:
 * - For i < (n+1)/2: returns i
 * - For i >= (n+1)/2: returns i - n
 *
 * @param i  Position in the FFT output (0 <= i < n)
 * @param n  Number of points in the transform
 * @return   Frequency index for position i
 */
inline Int fft_freqind(Index_t i, Index_t n) {
    Index_t half = (n + 1) / 2;  // Ceiling division
    if (i < half) {
        return static_cast<Int>(i);
    } else {
        return static_cast<Int>(i - n);
    }
}

/**
 * Compute the frequency bin indices for a full c2c FFT.
 *
 * For n samples, returns indices [0, 1, ..., n/2-1, -n/2, ..., -1] for even n,
 * or [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] for odd n.
 *
 * This is equivalent to numpy.fft.fftfreq(n) * n.
 *
 * @param n  Number of points in the transform
 * @return   Vector of integer frequency indices
 */
std::vector<Int> fft_freqind(Index_t n);

/**
 * Compute the frequency values for a full c2c FFT.
 *
 * Returns frequencies in cycles per unit spacing: f = k / (n * d)
 * where k is the frequency index from fft_freqind.
 *
 * This is equivalent to numpy.fft.fftfreq(n, d).
 *
 * @param n  Number of points in the transform
 * @param d  Sample spacing (default 1.0)
 * @return   Vector of frequency values
 */
std::vector<Real> fft_freq(Index_t n, Real d = 1.0);

/**
 * Compute the frequency bin indices for a r2c (half-complex) FFT.
 *
 * For n real input samples, the r2c transform produces n/2+1 complex outputs.
 * Returns indices [0, 1, ..., n/2].
 *
 * This is equivalent to numpy.fft.rfftfreq(n) * n.
 *
 * @param n  Number of real input points
 * @return   Vector of integer frequency indices (length n/2+1)
 */
std::vector<Int> rfft_freqind(Index_t n);

/**
 * Compute the frequency values for a r2c (half-complex) FFT.
 *
 * Returns frequencies in cycles per unit spacing: f = k / (n * d)
 * where k is the frequency index from rfft_freqind.
 *
 * This is equivalent to numpy.fft.rfftfreq(n, d).
 *
 * @param n  Number of real input points
 * @param d  Sample spacing (default 1.0)
 * @return   Vector of frequency values (length n/2+1)
 */
std::vector<Real> rfft_freq(Index_t n, Real d = 1.0);

/**
 * Compute the Fourier grid dimensions for a half-complex r2c transform.
 *
 * For a real-space grid of size [Nx, Ny, Nz], the half-complex Fourier grid
 * has size [Nx/2+1, Ny, Nz] (for the first axis being the r2c axis).
 *
 * @param nb_grid_pts  Real-space grid dimensions
 * @param r2c_axis     Axis along which r2c transform is performed (default 0)
 * @return             Fourier-space grid dimensions
 */
DynGridIndex get_hermitian_grid_pts(const DynGridIndex & nb_grid_pts,
                                  Index_t r2c_axis = 0);

/**
 * Compute the normalization factor for FFT roundtrip.
 *
 * For an unnormalized FFT, ifft(fft(x)) = N * x where N is the total number
 * of grid points. This returns 1/N for normalizing the result.
 *
 * @param nb_grid_pts  Grid dimensions
 * @return             Normalization factor (1.0 / total_grid_points)
 */
Real fft_normalization(const DynGridIndex & nb_grid_pts);

/**
 * Evenly distribute a global dimension across ranks.
 *
 * Returns the local size and offset for a given rank.
 *
 * @param global_size  Total size of the dimension
 * @param comm_size    Number of MPI ranks
 * @param rank         This rank's index (0-based)
 * @param local_size   Output: number of elements on this rank
 * @param offset       Output: starting index of this rank's portion
 */
void distribute_dimension(Index_t global_size, int comm_size, int rank,
                          Index_t & local_size, Index_t & offset);

/**
 * Select an optimal 2D process grid for P ranks.
 *
 * For P total ranks, finds P1 x P2 = P that minimizes communication.
 * Prefers P1 <= P2 and tries to make the grid as square as possible
 * while respecting the grid dimensions.
 *
 * @param num_ranks    Total number of MPI ranks
 * @param nb_grid_pts  Grid dimensions (used to prefer certain factorizations)
 * @param p1           Output: first dimension of process grid
 * @param p2           Output: second dimension of process grid
 */
void select_process_grid(int num_ranks, const DynGridIndex & nb_grid_pts, int & p1,
                         int & p2);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_UTILS_HH_
