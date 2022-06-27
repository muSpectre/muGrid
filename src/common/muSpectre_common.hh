/**
 * @file   muSpectre_common.hh
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
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#include <libmugrid/grid_common.hh>
#include <libmugrid/tensor_algebra.hh>
#include <libmugrid/exception.hh>

#include <libmufft/mufft_common.hh>

#include <string>

#ifndef SRC_COMMON_MUSPECTRE_COMMON_HH_
#define SRC_COMMON_MUSPECTRE_COMMON_HH_

namespace muSpectre {

  using muGrid::Dim_t;
  using muGrid::Index_t;

  using muGrid::Complex;
  using muGrid::Int;
  using muGrid::Real;
  using muGrid::Uint;

  using muGrid::oneD;
  using muGrid::twoD;
  using muGrid::threeD;

  using muGrid::OneQuadPt;
  using muGrid::TwoQuadPts;
  using muGrid::FourQuadPts;
  using muGrid::FiveQuadPts;
  using muGrid::SixQuadPts;
  using muGrid::EightQuadPts;

  using muGrid::OneNode;

  using muGrid::zerothOrder;
  using muGrid::firstOrder;
  using muGrid::secondOrder;
  using muGrid::fourthOrder;

  using muGrid::Ccoord_t;
  using muGrid::Rcoord_t;
  using muGrid::Shape_t;

  using muGrid::DynCcoord_t;
  using muGrid::DynRcoord_t;
  using muGrid::eigen;
  using muGrid::operator/;
  using muGrid::operator<<;

  using muGrid::apply;
  using muGrid::optional;

  using muGrid::get;
  using muGrid::T4Mat;
  using muGrid::T4MatMap;

  using muGrid::IterUnit;
  using muGrid::Mapping;

  using muGrid::PixelTag;
  const std::string QuadPtTag{"quad_point"};
  const std::string NodalPtTag{"nodal_point"};

  namespace Tensors = ::muGrid::Tensors;
  namespace Matrices = ::muGrid::Matrices;

  /**
   * Copyright banner to be printed to the terminal by executables
   * Arguments are the executable's name, year of writing and the name
   * + address of the copyright holder
   */
  void banner(std::string name, Uint year, std::string cpy_holder);

  namespace version {
    /**
     * returns a formatted text that can be printed to stdout or to output
     * files. It contains the git commit hash and repository url used to compile
     * µSpectre and whether the current state was dirty or not.
     */
    std::string info();
    const char * hash();
    const char * description();
    bool is_dirty();
  }  // namespace version

  //! continuum mechanics flags
  enum class Formulation {
    not_set,           //!< causes comput_stresses to throw an error
    finite_strain,     //!< causes evaluation in PK1(F)
    small_strain,      //!< causes evaluation in   σ(ε)
    small_strain_sym,  //!< symmetric storage as vector ε
    native  //! causes the material's native measures to be used in evaluation
  };

  enum class SolverType {
    Spectral,       //!< spectral framework, unknowns are strain and stress
    FiniteElements  //!< finite-element framework, unknows are displacement and
                    //!< force
  };

  //! split cell flags
  enum class SplitCell { laminate, simple, no };

  //! used to indicate whether internal (native) stresses should be stored
  //! (yes) or not stored (no)
  enum class StoreNativeStress { yes, no };

  /**
   * finite differences flags
   * NOTE: it is really important to notice that in situations that forward and
   * backward steps are different in nature (e.g. loading vs. unloading step on
   * the damage surface of a damage material), the centred option does not
   * correspond to any meaningful tangent of the material as it returns
   * an average of the forward and the backward schemes, which does not
   * represent the correct value of the tangent of in any case.
   */
  enum class FiniteDiff {
    forward,   //!< ∂f/∂x ≈ (f(x+Δx) - f(x))/Δx
    backward,  //!< ∂f/∂x ≈ (f(x) - f(x-Δx))/Δx
    centred,   //!< ∂f/∂x ≈ (f(x+Δx) - f(x-Δx))/2Δx
    outward,   //!< ∂f/∂x ≈ (f(x+sign(x)*Δx) - f(x))/(sign(x)*Δx)
    inward     //!< ∂f/∂x ≈ (f(x) - f(x-sign(x)*Δx))/(sign(x)*Δx)
  };

  /**
   * The Enum class determining the boundary condition
   */
  enum class MeanControl {
    StrainControl,  //!< strain boundary condition
    StressControl,  //!< stress boundary condition
    MixedControl    //!< mixed stress/strain boundary condition
  };
  /**
   * compile time computation of voigt vector
   */
  template <bool sym = true>
  constexpr Dim_t vsize(Dim_t dim) {
    if (sym) {
      return (dim * (dim - 1) / 2 + dim);
    } else {
      return dim * dim;
    }
  }

  //! compute the number of degrees of freedom to store for the strain
  //! tensor given dimension dim
  constexpr Dim_t dof_for_formulation(const Formulation form, const Dim_t dim,
                                      const Dim_t nb_quad_pts) {
    switch (form) {
    case Formulation::small_strain_sym:
      return vsize(dim) * nb_quad_pts;
    default:
      return muGrid::ipow(dim, 2) * nb_quad_pts;
    }
  }

  //! compute the shape of the strain tensor given dimension dim
  inline Shape_t shape_for_formulation(const Formulation form,
                                       const Dim_t dim) {
    switch (form) {
    case Formulation::small_strain_sym:
      return Shape_t({vsize(dim)});
    default:
      return Shape_t({dim, dim});
    }
  }

  //! compute the shape of the tangent tensor given dimension dim
  inline Shape_t t4shape_for_formulation(const Formulation form,
                                         const Dim_t dim) {
    switch (form) {
    case Formulation::small_strain_sym:
      return Shape_t({vsize(dim), vsize(dim)});
    default:
      return Shape_t({dim, dim, dim, dim});
    }
  }

  //! inserts `muSpectre::Formulation`s into `std::ostream`s
  std::ostream & operator<<(std::ostream & os, Formulation f);

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
  std::ostream & operator<<(std::ostream & os, StressMeasure s);

  /* ---------------------------------------------------------------------- */
  //! Material laws can declare which type of strain measure they require and
  //! µSpectre will provide it
  enum class StrainMeasure {
    PlacementGradient,     //!< placement gradient (δy/δx)
    DisplacementGradient,  //!< displacement gradient (δu/δx)
    Infinitesimal,         //!< small strain tensor .5(∇u + ∇uᵀ)
    GreenLagrange,   //!< Green-Lagrange strain .5(Fᵀ·F - I) = .5(U² - I)
    Biot,            //!< Biot strain (U - I and F = RU)
    Log,             //!< logarithmic strain (log U and F = RU)
    LogLeftStretch,  //!< logarithmic strain (log V and F = VR)
    Almansi,         //!< Almansi strain .5 (I - F⁻ᵀ. F⁻¹)
    RCauchyGreen,    //!< Right Cauchy-Green tensor (Fᵀ.F)
    LCauchyGreen,    //!< Left Cauchy-Green tensor(F.Fᵀ)
    no_strain_       //!< only for triggering static_assert
  };

  //! inserts `muSpectre::StrainMeasure`s into `std::ostream`s
  std::ostream & operator<<(std::ostream & os, StrainMeasure s);

  /* ---------------------------------------------------------------------- */
  /**
   * Returns either a bool expressing that whether a strain measure is objective
   * or not
   */

  constexpr bool is_objective(const StrainMeasure & measure) {
    // for the moment all the existing strain measures in the code are objective
    // except Gradient
    return (measure != StrainMeasure::PlacementGradient);
  }

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

  /** Compile-time function to g strain measure stored by muSpectre
      depending on the formulation
   **/
  constexpr StrainMeasure get_stored_strain_type(Formulation form,
                                                 SolverType solver_type) {
    switch (solver_type) {
    case SolverType::FiniteElements: {
      return StrainMeasure::DisplacementGradient;
      break;
    }
    case SolverType::Spectral: {
      switch (form) {
      case Formulation::small_strain: {
        return StrainMeasure::Infinitesimal;
        break;
      }
      case Formulation::finite_strain: {
        return StrainMeasure::PlacementGradient;
        break;
      }
      default:
        return StrainMeasure::no_strain_;
        break;
      }
    }
    default: {
      return StrainMeasure::no_strain_;
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
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
    case Formulation::native: {
      // fall-through
    }
    case Formulation::finite_strain: {
      return expected;
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

  //! gradient (or strain) shape for solvers. Always mapped onto a Matrix
  constexpr std::array<Index_t, 2>
  gradient_shape(Index_t rank, Index_t dim, bool is_mechanics = false,
                 Formulation formulation = Formulation::not_set) {
    if (is_mechanics) {
      if (rank != 2) {
        // dummy formulation is a workaround for GCC bug
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67371
        bool dummy{true};
        dummy ? throw muGrid::RuntimeError(
                    "Cannot determine gradient shape. I only "
                    "know how to handle rank-2 "
                    "mechanics problems")
              : false;
      }
      switch (formulation) {
      case Formulation::small_strain_sym: {
        return {vsize(dim), 1};
        break;
      }
      case Formulation::not_set: {
        bool dummy{true};
        dummy ? throw muGrid::RuntimeError(
                    "Cannot determinge gradient shape unless "
                    "the formulation has been set.")
              : false;
        return {0, 0};
        break;
      }
      default:
        return {dim, dim};
        break;
      }
    } else {
      switch (rank) {
      case 0: {
        return {1, 1};
        break;
      }
      case 1: {
        return {dim, 1};
        break;
      }
      case 2: {
        return {dim, dim};
        break;
      }
      default: {
        bool dummy{true};
        dummy ? throw muGrid::RuntimeError(
                    "Cannot determine gradient shape. I only "
                    "know how to handle ranks 0, 1, or 2")
              : false;
        return {0, 0};
        break;
      }
      }
    }
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_MUSPECTRE_COMMON_HH_
