/**
 * @file   grid_common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Small definitions of commonly used types throughout µgrid
 *
 * Copyright © 2019 Till Junge
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

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <complex>
#include <type_traits>

#ifndef SRC_LIBMUGRID_GRID_COMMON_HH_
#define SRC_LIBMUGRID_GRID_COMMON_HH_

namespace muGrid {

  /**
   * Eigen uses signed integers for dimensions. For consistency, µGrid uses them
   * througout the code. Needs to represent -1 for Eigen
   */
  using Dim_t = int;

  constexpr Dim_t oneD{1};         //!< constant for a one-dimensional problem
  constexpr Dim_t twoD{2};         //!< constant for a two-dimensional problem
  constexpr Dim_t threeD{3};       //!< constant for a three-dimensional problem
  constexpr Dim_t firstOrder{1};   //!< constant for vectors
  constexpr Dim_t secondOrder{2};  //!< constant second-order tensors
  constexpr Dim_t fourthOrder{4};  //!< constant fourth-order tensors

  //! \addtogroup Scalars Scalar types used for mathematical calculations
  //@{
  using Uint = unsigned int;
  using Int = int;
  using Real = double;
  using Complex = std::complex<Real>;
  //@}

  /**
   * Used to specify whether to iterate over pixels or quadrature points in
   * field maps
   */
  enum class Iteration { Pixel, QuadPt };

  /**
   * Maps can give constant or mutable access to the mapped field through their
   * iterators or access operators.
   */
  enum class Mapping { Const, Mut };

  //! \addtogroup Coordinates Coordinate types
  //@{
  //! Ccoord_t are cell coordinates, i.e. integer coordinates
  template <size_t Dim>
  using Ccoord_t = std::array<Dim_t, Dim>;
  //! Real space coordinates
  template <size_t Dim>
  using Rcoord_t = std::array<Real, Dim>;

  template<typename T, size_t Dim>
  Eigen::Map<Eigen::Matrix<T, Dim, 1>> eigen(std::array<T, Dim> & coord) {
    return Eigen::Map<Eigen::Matrix<T, Dim, 1>>{coord.data()};
  }
  template <typename T, size_t Dim>
  Eigen::Map<const Eigen::Matrix<T, Dim, 1>>
  eigen(const std::array<T, Dim> & coord) {
    return Eigen::Map<const Eigen::Matrix<T, Dim, 1>>{coord.data()};
  }
  //@}

  /**
   * Allows inserting `muGrid::Ccoord_t` and `muGrid::Rcoord_t`
   * into `std::ostream`s
   */
  template <typename T, size_t dim>
  std::ostream & operator<<(std::ostream & os,
                            const std::array<T, dim> & index) {
    os << "(";
    for (size_t i = 0; i < dim - 1; ++i) {
      os << index[i] << ", ";
    }
    os << index.back() << ")";
    return os;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Rcoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Ccoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! convenience definitions
  constexpr Real pi{3.1415926535897932384626433};
  //! constant used to explicitly denote unknown positive integers
  constexpr static Dim_t Unknown{-1};

  //! compile-time potentiation required for field-size computations
  template <typename R, typename I>
  constexpr R ipow(R base, I exponent) {
    static_assert(std::is_integral<I>::value, "Type must be integer");
    R retval{1};
    for (I i = 0; i < exponent; ++i) {
      retval *= base;
    }
    return retval;
  }
}  // namespace muGrid

#include "cpp_compliance.hh"

#endif  // SRC_LIBMUGRID_GRID_COMMON_HH_
