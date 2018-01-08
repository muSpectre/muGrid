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

#ifndef MATERIALS_TOOLBOX_H
#define MATERIALS_TOOLBOX_H

#include "common/common.hh"
#include "common/tensor_algebra.hh"
#include "common/eigen_tools.hh"
#include "common/T4_map_proxy.hh"

#include <Eigen/Dense>

#include <exception>
#include <sstream>
#include <iostream>
#include <tuple>
#include <type_traits>

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
    /**
     * Flag used to designate whether the material should compute both stress
     * and tangent moduli or only stress
     */
    enum class NeedTangent {
      yes, // compute both stress and tangent moduli
      no}; // compute only stress
    /**
     * Struct with a single variadic function only used to determine the
     * tuple type to store in the various sub-iterators or to return
     */

    template <MatTB::NeedTangent NeedTgt>
    struct StoredTuple {
      template <class ... Args>
      decltype(auto) operator()(Args && ... args) {
        return std::tie(args ... );
      }
    };

    /**
     * struct used to determine the exact type of a tuple of references obtained
     * when a bunch of iterators over fiel_maps are dereferenced and their
     * results are concatenated into a tuple
     */
    template <class... T>
    struct ReferenceTuple {
      using type = std::tuple<typename T::reference ...>;
    };

    /**
     * specialisation for tuples
     */
    //template <>
    template <class... T>
    struct ReferenceTuple<std::tuple<T...>> {
      using type = typename ReferenceTuple<T...>::type;
    };

    /**
     * helper type for ReferenceTuple
     */
    template <class... T>
    using ReferenceTuple_t = typename ReferenceTuple<T...>::type;

    /* ---------------------------------------------------------------------- */
    /** Structure for functions returning one strain measure as a
        function of another
     **/
    namespace internal {

      template <StrainMeasure In, StrainMeasure Out = In>
      struct ConvertStrain {
        static_assert((In == StrainMeasure::Gradient) ||
                      (In == StrainMeasure::Infinitesimal),
                      "This situation makes me suspect that you are not using "
                      "MatTb as intended. Disable this assert only if you are "
                      "sure about what you are doing.");

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
      struct ConvertStrain<StrainMeasure::Gradient, StrainMeasure::GreenLagrange> {

        template <class Strain_t>
        inline static decltype(auto)
        compute(Strain_t&& F) {
          return .5*(F.transpose()*F - Strain_t::PlainObject::Identity());

        }
      };
 
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

      template <Dim_t Dim,
                StressMeasure StressM,
                StrainMeasure StrainM>
      struct PK1_stress {

        template <class Strain_t, class Stress_t>
        inline static decltype(auto)
        compute(Strain_t && /*strain*/, Stress_t && /*stress*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                        (StressM == StressMeasure::PK1),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK2stress<PK1,T1, T2> for an example.");
        }

        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto)
        compute(Strain_t && /*strain*/, Stress_t && /*stress*/,
                Tangent_t && /*stiffness*/) {
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
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::PK1, StrainM>:
        public PK1_stress<Dim, StressMeasure::__nostress__,
                          StrainMeasure::__nostrain__> {

        template <class Strain_t, class Stress_t>
        inline static decltype(auto)
        compute(Strain_t && /*dummy*/, Stress_t && P) {
          return std::forward<Stress_t>(P);
        }
      };

      /* ---------------------------------------------------------------------- */
      /** Specialisation for the transparent case, where we already have PK1
          stress *and* stiffness is given with respect to the transformation
          gradient
       **/
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::PK1, StrainMeasure::Gradient>:
        public PK1_stress<Dim, StressMeasure::PK1,
                          StrainMeasure::__nostrain__> {

        using Parent = PK1_stress<Dim, StressMeasure::PK1,
                                  StrainMeasure::__nostrain__>;
        using Parent::compute;

        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto)
        compute(Strain_t && /*dummy*/, Stress_t && P, Tangent_t && K) {
          return std::make_tuple(std::forward<Stress_t>(P),
                                 std::forward<Tangent_t>(K));
        }
      };

      /* ---------------------------------------------------------------------- */
      /**
       * Specialisation for the case where we get material stress (PK2)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::PK2, StrainM>:
        public PK1_stress<Dim, StressMeasure::__nostress__,
                          StrainMeasure::__nostrain__> {

        template <class Strain_t, class Stress_t>
        inline static decltype(auto)
        compute(Strain_t && F, Stress_t && S) {
          return F*S;
        }
      };

      /* ---------------------------------------------------------------------- */
      /**
       * Specialisation for the case where we get material stress (PK2) derived
       * with respect to Green-Lagrange strain
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::PK2, StrainMeasure::GreenLagrange>:
        public PK1_stress<Dim, StressMeasure::PK2,
                          StrainMeasure::__nostrain__> {
        using Parent = PK1_stress<Dim, StressMeasure::PK2,
                                  StrainMeasure::__nostrain__>;
        using Parent::compute;

        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto)
        compute(Strain_t && F, Stress_t && S, Tangent_t && C) {
          using T4 = typename std::remove_reference_t<Tangent_t>::PlainObject;
          using Tmap = T4MatMap<Real, Dim>;
          T4 K;
          Tmap Kmap{K.data()};
          K.setZero();

          for (int i = 0; i < Dim; ++i) {
            for (int m = 0; m < Dim; ++m) {
              for (int n = 0; n < Dim; ++n) {
                get(Kmap,i,m,i,n) += S(m,n);
                for (int j = 0; j < Dim; ++j) {
                  for (int r = 0; r < Dim; ++r) {
                    for (int s = 0; s < Dim; ++s) {
                      get(Kmap,i,m,j,n) += F(i,r)*get(C,r,m,n,s)*(F(j,s));
                    }
                  }
                }
              }
            }
          }
          auto && P = compute(std::forward<Strain_t>(F),
                              std::forward<Stress_t>(S));
          return std::make_tuple(std::move(P), std::move(K));
        }
      };

    }  // internal
    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM,
              class Stress_t, class Strain_t>
    decltype(auto) PK1_stress(Strain_t && strain, Stress_t && stress) {
      constexpr Dim_t dim{EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      return internal::PK1_stress<dim, StressM, StrainM>::compute
        (std::forward<Strain_t>(strain),
         std::forward<Stress_t>(stress));
    };

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM,
              class Stress_t, class Strain_t, class Tangent_t>
    decltype(auto) PK1_stress(Strain_t  && strain,
                              Stress_t  && stress,
                              Tangent_t && tangent) {
      constexpr Dim_t dim{EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      static_assert((dim== EigenCheck::tensor_4_dim<Tangent_t>::value),
                    "Stress and tangent tensors have differing dimensions");
      return internal::PK1_stress<dim, StressM, StrainM>::compute
        (std::forward<Strain_t>(strain),
         std::forward<Stress_t>(stress),
         std::forward<Tangent_t>(tangent));
    };


  }  // MatTB

}  // muSpectre

#endif /* MATERIALS_TOOLBOX_H */
