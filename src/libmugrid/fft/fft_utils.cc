/**
 * @file   fft/fft_utils.cc
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

#include "fft_utils.hh"

#include <cmath>
#include <numeric>

namespace muGrid {

std::vector<Int> fft_freqind(Index_t n) {
  std::vector<Int> freq(n);

  // First half: 0, 1, 2, ..., n/2 (or (n-1)/2 for odd n)
  // Second half: -n/2, ..., -2, -1 (or -(n-1)/2, ..., -1 for odd n)
  Index_t half = (n + 1) / 2;  // Ceiling division

  for (Index_t i = 0; i < half; ++i) {
    freq[i] = static_cast<Int>(i);
  }

  for (Index_t i = half; i < n; ++i) {
    freq[i] = static_cast<Int>(i - n);
  }

  return freq;
}

std::vector<Real> fft_freq(Index_t n, Real d) {
  std::vector<Int> indices = fft_freqind(n);
  std::vector<Real> freq(n);

  Real scale = 1.0 / (n * d);
  for (Index_t i = 0; i < n; ++i) {
    freq[i] = indices[i] * scale;
  }

  return freq;
}

std::vector<Int> rfft_freqind(Index_t n) {
  Index_t nout = n / 2 + 1;
  std::vector<Int> freq(nout);

  for (Index_t i = 0; i < nout; ++i) {
    freq[i] = static_cast<Int>(i);
  }

  return freq;
}

std::vector<Real> rfft_freq(Index_t n, Real d) {
  std::vector<Int> indices = rfft_freqind(n);
  std::vector<Real> freq(indices.size());

  Real scale = 1.0 / (n * d);
  for (size_t i = 0; i < indices.size(); ++i) {
    freq[i] = indices[i] * scale;
  }

  return freq;
}

DynGridIndex get_hermitian_grid_pts(const DynGridIndex & nb_grid_pts,
                                  Index_t r2c_axis) {
  DynGridIndex result = nb_grid_pts;
  result[r2c_axis] = nb_grid_pts[r2c_axis] / 2 + 1;
  return result;
}

Real fft_normalization(const DynGridIndex & nb_grid_pts) {
  Index_t total = 1;
  for (Dim_t d = 0; d < nb_grid_pts.get_dim(); ++d) {
    total *= nb_grid_pts[d];
  }
  return 1.0 / total;
}

void distribute_dimension(Index_t global_size, int comm_size, int rank,
                          Index_t & local_size, Index_t & offset) {
  Index_t base_size = global_size / comm_size;
  Index_t remainder = global_size % comm_size;

  // First 'remainder' ranks get one extra element
  if (rank < remainder) {
    local_size = base_size + 1;
    offset = rank * (base_size + 1);
  } else {
    local_size = base_size;
    offset = remainder * (base_size + 1) + (rank - remainder) * base_size;
  }
}

void select_process_grid(int num_ranks, const DynGridIndex & nb_grid_pts, int & p1,
                         int & p2) {
  // For 2D grids, we just have one distribution dimension
  if (nb_grid_pts.get_dim() == 2) {
    p1 = 1;
    p2 = num_ranks;
    return;
  }

  // For 3D grids, find P1 x P2 = num_ranks that minimizes communication
  // Heuristic: prefer the most "square" factorization
  // Also prefer factorizations where grid dimensions are evenly divisible

  int best_p1 = 1;
  int best_p2 = num_ranks;
  double best_score = 0.0;

  // Get grid dimensions (for 3D: Y is distributed among P2, Z among P1)
  Index_t Ny = nb_grid_pts.get_dim() > 1 ? nb_grid_pts[1] : 1;
  Index_t Nz = nb_grid_pts.get_dim() > 2 ? nb_grid_pts[2] : 1;

  // Try all factorizations
  for (int test_p1 = 1; test_p1 <= num_ranks; ++test_p1) {
    if (num_ranks % test_p1 != 0) {
      continue;
    }
    int test_p2 = num_ranks / test_p1;

    // Compute a score based on:
    // 1. Squareness (higher is better)
    // 2. Divisibility of grid dimensions (higher is better)
    double squareness =
        1.0 -
        std::abs(static_cast<double>(test_p1 - test_p2)) / (test_p1 + test_p2);

    // Check divisibility
    double divisibility = 0.0;
    if (Nz % test_p1 == 0)
      divisibility += 1.0;
    if (Ny % test_p2 == 0)
      divisibility += 1.0;

    double score = squareness + 0.5 * divisibility;

    if (score > best_score) {
      best_score = score;
      best_p1 = test_p1;
      best_p2 = test_p2;
    }
  }

  p1 = best_p1;
  p2 = best_p2;
}

}  // namespace muGrid
