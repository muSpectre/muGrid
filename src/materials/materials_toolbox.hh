/**
 * file   materials_toolbox.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Nov 2017
 *
 * @brief  collection of common continuum mechanics tools
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

#ifndef MATERIALS_TOOLBOX_H
#define MATERIALS_TOOLBOX_H

#include <exception>
#include <sstream>
#include <iostream>
#include <tuple>
#include "common/common.hh"
#include "common/tensor_algebra.hh"

namespace muSpectre {

  namespace MatTB {
    class MaterialsToolboxError:public std::runtime_error{
    public:
      explicit MaterialsToolboxError(const std::string& what)
        :std::runtime_error(what){}
      explicit MaterialsToolboxError(const char * what)
        :std::runtime_error(what){}
    };


    /* ---------------------------------------------------------------------- */
    /** Structure for functions returning one strain measure as a
        function of another
     **/
    namespace internal {

      template <StrainMeasure In, StrainMeasure Out = In>
      struct ConvertStrain {
        template <class Strain_t>
        inline static decltype(auto)
        compute(Strain_t&& input) {
          // transparent case, in which no conversion is required:
          // just a perfect forwarding
          static_assert
            ((In == Out),
             "This particular strain conversion is not implemented");
          return std::forward<Strain_t>(input);
        }
      };

      /* ---------------------------------------------------------------------- */
      /** Specialisation for getting Green-Lagrange strain from the
          transformation gradient
      **/
      template <>
      template <class Strain_t>
      decltype(auto)
      ConvertStrain<StrainMeasure::Gradient, StrainMeasure::GreenLagrange>::
      compute(Strain_t && F) {
        return (F.transpose()*F - Strain_t::PlainObject::Identity());
      }

    }  // internal

    /* ---------------------------------------------------------------------- */
    //! set of functions returning one strain measure as a function of
    //! another
    template <StrainMeasure In, StrainMeasure Out,
              class Strain_t>
    decltype(auto) convert_strain(Strain_t && strain) {
      return internal::ConvertStrain<In, Out>::compute(std::move(strain));
    };



    /* ---------------------------------------------------------------------- */
    /** Structure for functions returning PK1 stress from other stress measures
     **/
    namespace internal {

      template <StressMeasure StressM,
                StrainMeasure StrainM = StrainMeasure::__nostrain__>
      struct PK1_stress {

        template <class Stress_t, class Strain_t>
        inline static decltype(auto)
        compute(Stress_t && /*stress*/, Strain_t && /*strain*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                        (StressM == StressMeasure::PK1),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK2stress<PK1,T1, T2> for an example.");
        }

        template <class Stress_t, class Strain_t, class Tangent_t>
        inline static decltype(auto)
        compute(Stress_t && /*stress*/, Tangent_t && /*stiffness*/,
                Strain_t && /*strain*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                        (StressM == StressMeasure::PK1),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK2stress<PK1,T1, T2> for an example.");
        }
      };

      /* ---------------------------------------------------------------------- */
      /** Specialisation for the transparent case, where we already
          have PK1 stress
       **/
      template <StrainMeasure StrainM>
      struct PK1_stress<StressMeasure::PK1, StrainM>:
        public PK1_stress<StressMeasure::__nostress__,
                          StrainMeasure::__nostrain__> {

        template <class Stress_t, class Strain_t>
        inline static decltype(auto)
        compute(Stress_t && P, Strain_t && /*dummy*/) {
          return std::forward<Stress_t>(P);
        }
      };

      /* ---------------------------------------------------------------------- */
      /** Specialisation for the transparent case, where we already have PK1
          stress *and* stiffness is given with respect to the transformation
          gradient
       **/
      template <>
      struct PK1_stress<StressMeasure::PK1, StrainMeasure::Gradient>:
        public PK1_stress<StressMeasure::PK1,
                          StrainMeasure::__nostrain__> {

        template <class Stress_t, class Strain_t, class Tangent_t>
        decltype(auto)
        compute(Stress_t && P, Tangent_t && K, Strain_t && /*dummy*/) {
          return std::forward_as_tuple(P, K);
        }
      };

      /* ---------------------------------------------------------------------- */
      /**
       * Specialisation for the case where we get material stress (PK2)
       */
      template <StrainMeasure StrainM>
      struct PK1_stress<StressMeasure::PK2, StrainM>:
        public PK1_stress<StressMeasure::__nostress__,
                          StrainMeasure::__nostrain__> {

        template <class Stress_t, class Strain_t>
        inline static decltype(auto)
        compute(Stress_t && S, Strain_t && F) {
          return F*S;
        }
      };

      /* ---------------------------------------------------------------------- */
      /**
       * Specialisation for the case where we get material stress (PK2) derived
       * with respect to Green-Lagrange strain
       */
      template <>
      struct PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>:
        public PK1_stress<StressMeasure::PK2,
                          StrainMeasure::__nostrain__> {

        template <class Stress_t, class Strain_t, class Tangent_t>
        inline static decltype(auto)
        compute(Stress_t && S, Tangent_t && C, Strain_t && F) {
          using T4 = typename Tangent_t::PlainObject;
          T4 K;
          K.setZero();
          constexpr int dim{Strain_t::ColsAtCompileTime};
          for (int i = 0; i < dim; ++i) {
            for (int m = 0; m < dim; ++m) {
              for (int n = 0; n < dim; ++n) {
                K(i,m,i,n) += S(m,n);
                for (int j = 0; j < dim; ++j) {
                  for (int r = 0; r < dim; ++r) {
                    for (int s = 0; s < dim; ++s) {
                      K(i,m,j,n) += F(i,r)*C(r,m,n,s)*(F(j,s));
                    }
                  }
                }
              }
            }
          }
          return K;
        }
      };

    }  // internal
    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM,
              class Stress_t, class Strain_t>
    decltype(auto) PK1_stress(Strain_t && strain, Stress_t && stress) {
      return internal::PK1_stress<StressM>::compute(std::move(stress),
                                                    std::move(strain));
    };

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM,
              class Stress_t, class Strain_t, class Tangent_t>
    decltype(auto) PK1_stress(Strain_t && strain,
                              Stress_t && stress,
                              Tangent_t && stiffness) {
      return internal::PK1_stress<StressM>::compute(std::move(stress),
                                                    std::move(strain),
                                                    std::move(stiffness));
    };

    /* ---------------------------------------------------------------------- */
    /** Structure for functions returning ∂strain/∂F (where F is the transformation
      * gradient. (e.g. for a material law expressed as PK2 stress S as
      * function of Green-Lagrange strain E, this would return F^T ⊗̲ I) (⊗_
      * refers to Curnier's notation) WARNING: this function assumes that your
      * stress measure has the two minor symmetries due to stress equilibrium
      * and the major symmetry due to the existance of an elastic potential
      * W(E). Do not rely on this function if you are using exotic stress
      * measures
     **/
    namespace internal {

      template<StrainMeasure StrainM>
      struct StrainDerivative
      {
        template <class Grad_t>
        inline static decltype(auto)
        compute (Grad_t && grad) {
          // the following test always fails to generate a compile-time error
          static_assert((StrainM == StrainMeasure::Almansi) &&
                        (StrainM == StrainMeasure::Biot),
                        "The requested StrainDerivative calculation is not "
                        "implemented. You either made a programming mistake or "
                        "need to implement it as a specialisation of this "
                        "function. See "
                        "StrainDerivative<StrainMeasure::GreenLagrange> for an "
                        "example.");
        }
      };

      template <>
      template <class Grad_t>
      decltype(auto) StrainDerivative<StrainMeasure::GreenLagrange>::
      compute(Grad_t && grad) {
      }

    }  // internal


    /* ---------------------------------------------------------------------- */
    //! set of functions returning ∂strain/∂F
    template<StrainMeasure StrainM>
    auto StrainDerivative = [](auto && F) decltype(auto) {
      // the following test always fails to generate a compile-time error
      static_assert((StrainM == StrainMeasure::Almansi) &&
                    (StrainM == StrainMeasure::Biot),
                    "The requested StrainDerivative calculation is not "
                    "implemented. You either made a programming mistake or "
                    "need to implement it as a specialisation of this "
                    "function. See "
                    "StrainDerivative<StrainMeasure::GreenLagrange> for an "
                    "example.");
    };

    /* ---------------------------------------------------------------------- */
    template<>
    auto StrainDerivative<StrainMeasure::GreenLagrange> =
      [] (auto && F) -> decltype(auto) {
      constexpr size_t order{2};
      constexpr Dim_t dim{F.Dimensions[0]};
      return Tensors::outer_under<dim>
        (F.shuffle(std::array<Dim_t, order>{1,0}), Tensors::I2<dim>());
    };

  }  // MatTB

}  // muSpectre

#endif /* MATERIALS_TOOLBOX_H */
