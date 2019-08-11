/**
 * @file   mpi_test_projection.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   16 Jan 2018
 *
 * @brief  common declarations for testing both the small and finite strain
 *         projection operators
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "mpi_context.hh"

#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

#ifndef TESTS_MPI_TEST_PROJECTION_HH_
#define TESTS_MPI_TEST_PROJECTION_HH_

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  struct Sizes {};
  template <>
  struct Sizes<twoD> {
    constexpr static Ccoord_t<twoD> get_nb_grid_pts() {
      return Ccoord_t<twoD>{3, 5};
    }
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{3.4, 5.8};
    }
  };
  template <>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_nb_grid_pts() {
      return Ccoord_t<threeD>{3, 5, 7};
    }
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{3.4, 5.8, 6.7};
    }
  };
  template <Dim_t DimS>
  struct Squares {};
  template <>
  struct Squares<twoD> {
    constexpr static Ccoord_t<twoD> get_nb_grid_pts() {
      return Ccoord_t<twoD>{5, 5};
    }
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{5, 5};
    }
  };
  template <>
  struct Squares<threeD> {
    constexpr static Ccoord_t<threeD> get_nb_grid_pts() {
      return Ccoord_t<threeD>{7, 7, 7};
    }
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{7, 7, 7};
    }
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class SizeGiver, class Proj, class Engine,
            bool parallel = true>
  struct ProjectionFixture {
    using Parent = Proj;
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    constexpr static bool is_parallel{parallel};
    ProjectionFixture()
        : projector(std::make_unique<Engine>(SizeGiver::get_nb_grid_pts(),
                                             muGrid::ipow(mdim, 2),
                                             MPIContext::get_context().comm),
                    SizeGiver::get_lengths()) {}
    Parent projector;
  };

}  // namespace muFFT

#endif  // TESTS_MPI_TEST_PROJECTION_HH_
