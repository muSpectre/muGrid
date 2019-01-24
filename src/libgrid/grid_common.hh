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
 * µgrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µgrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µgrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */


#ifndef SRC_MUGRID_GRID_COMMON_HH_
#define SRC_MUGRID_GRID_COMMON_HH_

namespace muGrid {

  /**
   * Eigen uses signed integers for dimensions. For consistency,
   µSpectre uses them througout the code. needs to represent -1 for
   eigen
   */
  using Dim_t = int;

  constexpr Dim_t oneD{1};         //!< constant for a one-dimensional problem
  constexpr Dim_t twoD{2};         //!< constant for a two-dimensional problem
  constexpr Dim_t threeD{3};       //!< constant for a three-dimensional problem
  constexpr Dim_t firstOrder{1};   //!< constant for vectors
  constexpr Dim_t secondOrder{2};  //!< constant second-order tensors
  constexpr Dim_t fourthOrder{4};  //!< constant fourth-order tensors

  //@{
  //! @anchor scalars
  //! Scalar types used for mathematical calculations
  using Uint = unsigned int;
  using Int = int;
  using Real = double;
  using Complex = std::complex<Real>;
  //@}

  //! Ccoord_t are cell coordinates, i.e. integer coordinates
  template <Dim_t dim>
  using Ccoord_t = std::array<Dim_t, dim>;
  //! Real space coordinates
  template <Dim_t dim>
  using Rcoord_t = std::array<Real, dim>;

  /**
   * Allows inserting `muSpectre::Ccoord_t` and `muSpectre::Rcoord_t`
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

  //! compile-time potentiation required for field-size computations
  template <typename R, typename I>
  constexpr R ipow(R base, I exponent) {
    static_assert(std::is_integral<I>::value, "Type must be integer");
    R retval{1};
    for (I i = 0; i < exponent; ++i) {
      retval *= base;
    }
    return retval;

  }  // muGrid

#endif  // SRC_MUGRID_GRID_COMMON_HH_
