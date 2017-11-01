/**
 * file   common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 May 2017
 *
 * @brief  Small definitions of commonly used types througout µSpectre
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <array>
#include <complex>
#include <iostream>
#include <type_traits>

#ifndef COMMON_H
#define COMMON_H


namespace muSpectre {

  //! Eigen uses signed integers for dimensions. For consistency, µSpectre uses
  //! them througout the code
  using Dim_t = int;// needs to represent -1 for eigen
  const Dim_t oneD{1};
  const Dim_t twoD{2};
  const Dim_t threeD{3};

  //! Ccoord_t are cell coordinates, i.e. integer coordinates
  template<Dim_t dim>
  using Ccoord_t = std::array<Dim_t, dim>;

  template<size_t dim>
  std::ostream & operator << (std::ostream & os, const Ccoord_t<dim> & index) {
    os << "(";
    for (size_t i = 0; i < dim-1; ++i) {
      os << index[i] << ", ";
    }
    os << index.back() << ")";
    return os;
  }

  //! Scalar types used for mathematical calculations
  using Uint = unsigned int;
  using Int = int;
  using Real = double;
  using Complex = std::complex<Real>;

  //! compile-time potentiation required for field-size computations
  template <typename I>
  constexpr I ipow(I base, I exponent) {
    static_assert(std::is_integral<I>::value, "Type must be integer");
    I retval{1};
    for (I i = 0; i < exponent; ++i) {
      retval *= base;
    }
    return retval;
  }

  //! continuum mechanics flags
  enum class Formulation{finite_strain, small_strain};


}  // muSpectre

#endif /* COMMON_H */
