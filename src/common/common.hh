/**
 * @file   common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 May 2017
 *
 * @brief  Small definitions of commonly used types throughout µSpectre
 *
 * @section  LICENSE
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
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
 */

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

#ifndef SRC_COMMON_COMMON_HH_
#define SRC_COMMON_COMMON_HH_

namespace muSpectre {

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
  template <Dim_t dim> using Ccoord_t = std::array<Dim_t, dim>;
  //! Real space coordinates
  template <Dim_t dim> using Rcoord_t = std::array<Real, dim>;

  /**
   * Allows inserting `muSpectre::Ccoord_t` and `muSpectre::Rcoord_t`
   * into `std::ostream`s
   */
  template <typename T, size_t dim>
  std::ostream &operator<<(std::ostream &os, const std::array<T, dim> &index) {
    os << "(";
    for (size_t i = 0; i < dim - 1; ++i) {
      os << index[i] << ", ";
    }
    os << index.back() << ")";
    return os;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> &a, const Rcoord_t<dim> &b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> &a, const Ccoord_t<dim> &b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! convenience definitions
  constexpr Real pi{3.1415926535897932384626433};

  //! compile-time potentiation required for field-size computations
  template <typename R, typename I> constexpr R ipow(R base, I exponent) {
    static_assert(std::is_integral<I>::value, "Type must be integer");
    R retval{1};
    for (I i = 0; i < exponent; ++i) {
      retval *= base;
    }
    return retval;
  }

  /**
   * Copyright banner to be printed to the terminal by executables
   * Arguments are the executable's name, year of writing and the name
   * + address of the copyright holder
   */
  void banner(std::string name, Uint year, std::string cpy_holder);

  /**
   * Planner flags for FFT (follows FFTW, hopefully this choice will
   * be compatible with alternative FFT implementations)
   * @enum muSpectre::FFT_PlanFlags
   */
  enum class FFT_PlanFlags {
    estimate,  //!< cheapest plan for slowest execution
    measure,   //!< more expensive plan for fast execution
    patient    //!< very expensive plan for fastest execution
  };

  //! continuum mechanics flags
  enum class Formulation {
    finite_strain,    //!< causes evaluation in PK1(F)
    small_strain,     //!< causes evaluation in   σ(ε)
    small_strain_sym  //!< symmetric storage as vector ε
  };

  /**
   * compile time computation of voigt vector
   */
  template <bool sym = true> constexpr Dim_t vsize(Dim_t dim) {
    if (sym) {
      return (dim * (dim - 1) / 2 + dim);
    } else {
      return dim * dim;
    }
  }

  //! compute the number of degrees of freedom to store for the strain
  //! tenor given dimension dim
  constexpr Dim_t dof_for_formulation(const Formulation form, const Dim_t dim) {
    switch (form) {
    case Formulation::small_strain_sym: {
      return vsize(dim);
      break;
    }
    default:
      return ipow(dim, 2);
      break;
    }
  }

  //! inserts `muSpectre::Formulation`s into `std::ostream`s
  std::ostream &operator<<(std::ostream &os, Formulation f);

  /* ---------------------------------------------------------------------- */
  //! Material laws can declare which type of stress measure they provide,
  //! and µSpectre will handle conversions
  enum class StressMeasure {
    Cauchy,     //!< Cauchy stress σ
    PK1,        //!< First Piola-Kirchhoff stress
    PK2,        //!< Second Piola-Kirchhoff stress
    Kirchhoff,  //!< Kirchhoff stress τ
    Biot,       //!< Biot stress
    Mandel,     //!< Mandel stress
    no_stress_  //!< only for triggering static_asserts
  };
  //! inserts `muSpectre::StressMeasure`s into `std::ostream`s
  std::ostream &operator<<(std::ostream &os, StressMeasure s);

  /* ---------------------------------------------------------------------- */
  //! Material laws can declare which type of strain measure they require and
  //! µSpectre will provide it
  enum class StrainMeasure {
    Gradient,       //!< placement gradient (δy/δx)
    Infinitesimal,  //!< small strain tensor .5(∇u + ∇uᵀ)
    GreenLagrange,  //!< Green-Lagrange strain .5(Fᵀ·F - I)
    Biot,           //!< Biot strain
    Log,            //!< logarithmic strain
    Almansi,        //!< Almansi strain
    RCauchyGreen,   //!< Right Cauchy-Green tensor
    LCauchyGreen,   //!< Left Cauchy-Green tensor
    no_strain_      //!< only for triggering static_assert
  };
  //! inserts `muSpectre::StrainMeasure`s into `std::ostream`s
  std::ostream &operator<<(std::ostream &os, StrainMeasure s);

  /* ---------------------------------------------------------------------- */
  /**
   * all isotropic elastic moduli to identify conversions, such as E
   * = µ(3λ + 2µ)/(λ+µ). For the full description, see
   * https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
   * Not all the conversions are implemented, so please add as needed
   */
  enum class ElasticModulus {
    Bulk,          //!< Bulk modulus K
    K = Bulk,      //!< alias for ``ElasticModulus::Bulk``
    Young,         //!< Young's modulus E
    E = Young,     //!< alias for ``ElasticModulus::Young``
    lambda,        //!< Lamé's first parameter λ
    Shear,         //!< Shear modulus G or µ
    G = Shear,     //!< alias for ``ElasticModulus::Shear``
    mu = Shear,    //!< alias for ``ElasticModulus::Shear``
    Poisson,       //!< Poisson's ratio ν
    nu = Poisson,  //!< alias for ``ElasticModulus::Poisson``
    Pwave,         //!< P-wave modulus M
    M = Pwave,     //!< alias for ``ElasticModulus::Pwave``
    no_modulus_
  };  //!< only for triggering static_asserts

  /**
   * define comparison in order to exploit that moduli can be
   * expressed in terms of any two other moduli in any order (e.g. K
   * = K(E, ν) = K(ν, E)
   */
  constexpr inline bool operator<(ElasticModulus A, ElasticModulus B) {
    return static_cast<int>(A) < static_cast<int>(B);
  }
  /* ---------------------------------------------------------------------- */
  /** Compile-time function to g strain measure stored by muSpectre
      depending on the formulation
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

  /** Compile-time function to g stress measure stored by muSpectre
      depending on the formulation
   **/
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

}  // namespace muSpectre

#ifndef EXPLICITLY_TURNED_ON_CXX17
#include "common/utilities.hh"
#endif

#endif  // SRC_COMMON_COMMON_HH_
