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

#include <string>
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
 * For n real input samples, the r2c transform produces n/2+1 complex outputs
 * (the remaining modes follow from Hermitian symmetry
 * F(-k) = conj(F(k))). Returns indices [0, 1, ..., n/2] (integer division).
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
 * The transforms are unnormalised, so ifft(fft(x)) = N x with
 * N = prod_alpha N_alpha the total number of grid points. This returns 1/N.
 *
 * @param nb_grid_pts  Grid dimensions
 * @return             Normalization factor (1.0 / total_grid_points)
 */
Real fft_normalization(const DynGridIndex & nb_grid_pts);

/**
 * Evenly distribute a global dimension across ranks.
 *
 * With b = global_size / comm_size and r = global_size mod comm_size, the
 * first r ranks get b + 1 elements and the others b, so local sizes differ
 * by at most one and the blocks are contiguous in rank order.
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
 * Select the 2D process grid P1 x P2 = num_ranks for the FFT decomposition.
 *
 * Real space is split as Y across P2 and Z across P1 (X is never split);
 * Fourier space as the half-complex X across P2 and Y across P1.
 *
 * - 1D: only num_ranks = 1 is supported (throws otherwise); P1 = P2 = 1.
 * - 2D: P1 = 1, P2 = num_ranks (a slab along Y; all modes coincide).
 * - 3D "slab": P1 = num_ranks, P2 = 1, so the (X, Y) plane stays local and
 *   is transformed by one N-D transform followed by a single Y<->Z
 *   transpose. Throws if num_ranks > Nz.
 * - 3D "auto": slab whenever num_ranks <= Nz (one transpose instead of two),
 *   otherwise the "pencil" heuristic.
 * - 3D "pencil": over all factorizations, maximise
 *   score = squareness + 0.5 * divisibility + 10 * fits, with
 *   squareness = 1 - |P1 - P2| / (P1 + P2), divisibility counting
 *   Nz mod P1 = 0 and Ny mod P2 = 0, and fits = (P1 <= min(Nz, Ny) and
 *   P2 <= min(Ny, Nx/2+1)), i.e. no rank is left without data. Ties go to
 *   the smaller P1. Cost O(num_ranks).
 *
 * @param num_ranks    Total number of MPI ranks
 * @param nb_grid_pts  Global grid dimensions (enter the divisibility and fit
 *                     terms of the score)
 * @param p1           Output: first dimension of process grid (splits Z in
 *                     real space, Y in Fourier space)
 * @param p2           Output: second dimension of process grid (splits Y in
 *                     real space, half-complex X in Fourier space)
 * @param decomposition  "auto" (default), "pencil" or "slab", as above.
 *                       Any other value throws.
 */
void select_process_grid(int num_ranks, const DynGridIndex & nb_grid_pts, int & p1,
                         int & p2, const std::string & decomposition = "auto");

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_UTILS_HH_
