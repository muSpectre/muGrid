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
 * Copyright © 2017 Till Junge
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
#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

#ifndef COMMON_H
#define COMMON_H


namespace muSpectre {

  //! Eigen uses signed integers for dimensions. For consistency, µSpectre uses
  //! them througout the code
  using Dim_t = int;// needs to represent -1 for eigen
  constexpr Dim_t oneD{1};
  constexpr Dim_t twoD{2};
  constexpr Dim_t threeD{3};
  constexpr Dim_t firstOrder{1};
  constexpr Dim_t secondOrder{2};
  constexpr Dim_t fourthOrder{4};

  //! Scalar types used for mathematical calculations
  using Uint = unsigned int;
  using Int = int;
  using Real = double;
  using Complex = std::complex<Real>;

  //! Ccoord_t are cell coordinates, i.e. integer coordinates
  template<Dim_t dim>
  using Ccoord_t = std::array<Dim_t, dim>;
  //! Real space coordinates
  template<Dim_t dim>
  using Rcoord_t = std::array<Real, dim>;

  template<typename T, size_t dim>
  std::ostream & operator << (std::ostream & os,
                              const std::array<T, dim> & index) {
    os << "(";
    for (size_t i = 0; i < dim-1; ++i) {
      os << index[i] << ", ";
    }
    os << index.back() << ")";
    return os;
  }

  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Rcoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i]/=b[i];
    }
    return retval;
  }

  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Ccoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i]/=b[i];
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
  }

  //! continuum mechanics flags
  enum class Formulation{finite_strain, small_strain};
  std::ostream & operator<<(std::ostream & os, Formulation f);

  /* ---------------------------------------------------------------------- */
  //! Material laws can declare which type of stress measure they provide,
  //! and µSpectre will handle conversions
  enum class StressMeasure {
    Cauchy, PK1, PK2, Kirchhoff, Biot, Mandel, no_stress_};
  std::ostream & operator<<(std::ostream & os, StressMeasure s);

  /* ---------------------------------------------------------------------- */
  //! Material laws can declare which type of strain measure they require and
  //! µSpectre will provide it
  enum class StrainMeasure {
    Gradient, Infinitesimal, GreenLagrange, Biot, Log, Almansi,
    RCauchyGreen, LCauchyGreen, no_strain_};
  std::ostream & operator<<(std::ostream & os, StrainMeasure s);

  /* ---------------------------------------------------------------------- */
  /** Compile-time functions to set the stress and strain measures
      stored by mu_spectre depending on the formulation
   **/
  constexpr StrainMeasure get_stored_strain_type(Formulation form) {
    switch (form) {
    case Formulation::finite_strain: {
      return StrainMeasure::Gradient;
      break;
    }
    case Formulation::small_strain: {
      return StrainMeasure::Infinitesimal;
      break;
    }
    default:
      return StrainMeasure::no_strain_;
      break;
    }
  }
  constexpr StressMeasure get_stored_stress_type(Formulation form) {
    switch (form) {
    case Formulation::finite_strain: {
      return StressMeasure::PK1;
      break;
    }
    case Formulation::small_strain: {
      return StressMeasure::Cauchy;
      break;
    }
    default:
      return StressMeasure::no_stress_;
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  /** Compile-time functions to get the stress and strain measures
      after they may have been modified by choosing a formulation.

      For instance, a law that expecs a Green-Lagrange strain as input
      will get the infinitesimal strain tensor instead in a small
      strain computation
   **/
  constexpr StrainMeasure get_formulation_strain_type(Formulation form,
                                                      StrainMeasure expected) {
    switch (form) {
    case Formulation::finite_strain: {
      return expected;
      break;
    }
    case Formulation::small_strain: {
      return get_stored_strain_type(form);
      break;
    }
    default:
      return StrainMeasure::no_strain_;
      break;
    }
  }

}  // muSpectre


#ifndef EXPLICITLY_TURNED_ON_CXX17
#include "common/utilities.hh"
#endif

#endif /* COMMON_H */
