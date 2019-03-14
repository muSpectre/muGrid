/**
 * @file   materials_toolbox.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Nov 2017
 *
 * @brief  collection of common continuum mechanics tools
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

#ifndef SRC_MATERIALS_MATERIALS_TOOLBOX_HH_
#define SRC_MATERIALS_MATERIALS_TOOLBOX_HH_

#include "common/muSpectre_common.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/T4_map_proxy.hh>
#include <libmugrid/tensor_algebra.hh>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <exception>
#include <sstream>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace muSpectre {

  namespace MatTB {
    /**
     * thrown when generic materials-related runtime errors occur
     * (mostly continuum mechanics problems)
     */
    class MaterialsToolboxError : public std::runtime_error {
     public:
      //! constructor
      explicit MaterialsToolboxError(const std::string & what)
          : std::runtime_error(what) {}
      //! constructor
      explicit MaterialsToolboxError(const char * what)
          : std::runtime_error(what) {}
    };

    /* ---------------------------------------------------------------------- */
    /**
     * Flag used to designate whether the material should compute both stress
     * and tangent moduli or only stress
     */
    enum class NeedTangent {
      yes,  //!< compute both stress and tangent moduli
      no    //!< compute only stress
    };

    /**
     * struct used to determine the exact type of a tuple of references obtained
     * when a bunch of iterators over fiel_maps are dereferenced and their
     * results are concatenated into a tuple
     */
    template <class... T>
    struct ReferenceTuple {
      //! use this type
      using type = std::tuple<typename T::reference...>;
    };

    /**
     * specialisation for tuples
     */
    // template <>
    template <class... T>
    struct ReferenceTuple<std::tuple<T...>> {
      //! use this type
      using type = typename ReferenceTuple<T...>::type;
    };

    /**
     * helper type for ReferenceTuple
     */
    template <class... T>
    using ReferenceTuple_t = typename ReferenceTuple<T...>::type;

    /* ---------------------------------------------------------------------- */
    namespace internal {

      /** Structure for functions returning one strain measure as a
       *  function of another
       **/
      template <StrainMeasure In, StrainMeasure Out = In>
      struct ConvertStrain {
        static_assert((In == StrainMeasure::Gradient) ||
                          (In == StrainMeasure::Infinitesimal),
                      "This situation makes me suspect that you are not using "
                      "MatTb as intended. Disable this assert only if you are "
                      "sure about what you are doing.");

        //! returns the converted strain
        template <class Strain_t>
        inline static decltype(auto) compute(Strain_t && input) {
          // transparent case, in which no conversion is required:
          // just a perfect forwarding
          static_assert((In == Out),
                        "This particular strain conversion is not implemented");
          return std::forward<Strain_t>(input);
        }
      };

      /* ----------------------------------------------------------------------
       */
      /** Specialisation for getting Green-Lagrange strain from the
          transformation gradient
          E = ¹/₂ (C - I) = ¹/₂ (Fᵀ·F - I)
      **/
      template <>
      struct ConvertStrain<StrainMeasure::Gradient,
                           StrainMeasure::GreenLagrange> {
        //! returns the converted strain
        template <class Strain_t>
        inline static decltype(auto) compute(Strain_t && F) {
          return .5 * (F.transpose() * F - Strain_t::PlainObject::Identity());
        }
      };

      /* ----------------------------------------------------------------------
       */
      /** Specialisation for getting Left Cauchy-Green strain from the
          transformation gradient
          B = F·Fᵀ = V²
      **/
      template <>
      struct ConvertStrain<StrainMeasure::Gradient,
                           StrainMeasure::LCauchyGreen> {
        //! returns the converted strain
        template <class Strain_t>
        inline static decltype(auto) compute(Strain_t && F) {
          return F * F.transpose();
        }
      };

      /* ----------------------------------------------------------------------
       */
      /** Specialisation for getting Right Cauchy-Green strain from the
          transformation gradient
          C = Fᵀ·F = U²
      **/
      template <>
      struct ConvertStrain<StrainMeasure::Gradient,
                           StrainMeasure::RCauchyGreen> {
        //! returns the converted strain
        template <class Strain_t>
        inline static decltype(auto) compute(Strain_t && F) {
          return F.transpose() * F;
        }
      };

      /* ----------------------------------------------------------------------
       */
      /** Specialisation for getting logarithmic (Hencky) strain from the
          transformation gradient
          E₀ = ¹/₂ ln C = ¹/₂ ln (Fᵀ·F)
      **/
      template <>
      struct ConvertStrain<StrainMeasure::Gradient, StrainMeasure::Log> {
        //! returns the converted strain
        template <class Strain_t>
        inline static decltype(auto) compute(Strain_t && F) {
          constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
          return (.5 * muGrid::logm(
                           Eigen::Matrix<Real, dim, dim>{F.transpose() * F}))
              .eval();
        }
      };

    }  // namespace internal

    /* ---------------------------------------------------------------------- */
    //! set of functions returning one strain measure as a function of
    //! another
    template <StrainMeasure In, StrainMeasure Out, class Strain_t>
    decltype(auto) convert_strain(Strain_t && strain) {
      return internal::ConvertStrain<In, Out>::compute(std::move(strain));
    }

    namespace internal {

      //! Base template for elastic modulus conversion
      template <ElasticModulus Out, ElasticModulus In1, ElasticModulus In2>
      struct Converter {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & /*in1*/,
                                             const Real & /*in2*/) {
          // if no return has happened until now, the conversion is not
          // implemented yet
          static_assert(
              (In1 == In2),
              "This conversion has not been implemented yet, please add "
              "it here below as a specialisation of this function "
              "template. Check "
              "https://en.wikipedia.org/wiki/Lam%C3%A9_parameters for "
              "the formula.");
          return 0;
        }
      };

      /**
       * Spectialisation for when the output is the first input
       */
      template <ElasticModulus Out, ElasticModulus In>
      struct Converter<Out, Out, In> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & A,
                                             const Real & /*B*/) {
          return A;
        }
      };

      /**
       * Spectialisation for when the output is the second input
       */
      template <ElasticModulus Out, ElasticModulus In>
      struct Converter<Out, In, Out> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & /*A*/,
                                             const Real & B) {
          return B;
        }
      };

      /**
       * Specialisation μ(E, ν)
       */
      template <>
      struct Converter<ElasticModulus::Shear, ElasticModulus::Young,
                       ElasticModulus::Poisson> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & E, const Real & nu) {
          return E / (2 * (1 + nu));
        }
      };

      /**
       * Specialisation λ(E, ν)
       */
      template <>
      struct Converter<ElasticModulus::lambda, ElasticModulus::Young,
                       ElasticModulus::Poisson> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & E, const Real & nu) {
          return E * nu / ((1 + nu) * (1 - 2 * nu));
        }
      };

      /**
       * Specialisation K(E, ν)
       */
      template <>
      struct Converter<ElasticModulus::Bulk, ElasticModulus::Young,
                       ElasticModulus::Poisson> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & E, const Real & nu) {
          return E / (3 * (1 - 2 * nu));
        }
      };

      /**
       * Specialisation E(K, µ)
       */
      template <>
      struct Converter<ElasticModulus::Young, ElasticModulus::Bulk,
                       ElasticModulus::Shear> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & K, const Real & G) {
          return 9 * K * G / (3 * K + G);
        }
      };

      /**
       * Specialisation ν(K, µ)
       */
      template <>
      struct Converter<ElasticModulus::Poisson, ElasticModulus::Bulk,
                       ElasticModulus::Shear> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & K, const Real & G) {
          return (3 * K - 2 * G) / (2 * (3 * K + G));
        }
      };

      /**
       * Specialisation E(λ, µ)
       */
      template <>
      struct Converter<ElasticModulus::Young, ElasticModulus::lambda,
                       ElasticModulus::Shear> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & lambda,
                                             const Real & G) {
          return G * (3 * lambda + 2 * G) / (lambda + G);
        }
      };

      /**
       * Specialisation λ(K, µ)
       */
      template <>
      struct Converter<ElasticModulus::lambda, ElasticModulus::Bulk,
                       ElasticModulus::Shear> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & K, const Real & mu) {
          return K - 2. * mu / 3.;
        }
      };

      /**
       * Specialisation K(λ, µ)
       */
      template <>
      struct Converter<ElasticModulus::Bulk, ElasticModulus::lambda,
                       ElasticModulus::Shear> {
        //! wrapped function (raison d'être)
        inline constexpr static Real compute(const Real & lambda,
                                             const Real & G) {
          return lambda + (2 * G) / 3;
        }
      };

    }  // namespace internal

    /**
     * allows the conversion from any two distinct input moduli to a
     * chosen output modulus
     */
    template <ElasticModulus Out, ElasticModulus In1, ElasticModulus In2>
    inline constexpr Real convert_elastic_modulus(const Real & in1,
                                                  const Real & in2) {
      // enforcing sanity
      static_assert((In1 != In2),
                    "The input modulus types cannot be identical");

      // enforcing independence from order in which moduli are supplied
      constexpr bool inverted{In1 > In2};
      using Converter =
          std::conditional_t<inverted, internal::Converter<Out, In2, In1>,
                             internal::Converter<Out, In1, In2>>;
      if (inverted) {
        return Converter::compute(std::move(in2), std::move(in1));
      } else {
        return Converter::compute(std::move(in1), std::move(in2));
      }
    }

    //! static inline implementation of Hooke's law
    template <Dim_t Dim, class Strain_t, class Tangent_t>
    struct Hooke {
      /**
       * compute Lamé's first constant
       * @param young: Young's modulus
       * @param poisson: Poisson's ratio
       */
      inline static constexpr Real compute_lambda(const Real & young,
                                                  const Real & poisson) {
        return convert_elastic_modulus<ElasticModulus::lambda,
                                       ElasticModulus::Young,
                                       ElasticModulus::Poisson>(young, poisson);
      }

      /**
       * compute Lamé's second constant (i.e., shear modulus)
       * @param young: Young's modulus
       * @param poisson: Poisson's ratio
       */
      inline static constexpr Real compute_mu(const Real & young,
                                              const Real & poisson) {
        return convert_elastic_modulus<ElasticModulus::Shear,
                                       ElasticModulus::Young,
                                       ElasticModulus::Poisson>(young, poisson);
      }

      /**
       * compute the bulk modulus
       * @param young: Young's modulus
       * @param poisson: Poisson's ratio
       */
      inline static constexpr Real compute_K(const Real & young,
                                             const Real & poisson) {
        return convert_elastic_modulus<ElasticModulus::Bulk,
                                       ElasticModulus::Young,
                                       ElasticModulus::Poisson>(young, poisson);
      }

      /**
       * compute the stiffness tensor
       * @param lambda: Lamé's first constant
       * @param mu: Lamé's second constant (i.e., shear modulus)
       */
      inline static Eigen::TensorFixedSize<Real,
                                           Eigen::Sizes<Dim, Dim, Dim, Dim>>
      compute_C(const Real & lambda, const Real & mu) {
        return lambda *
                   Tensors::outer<Dim>(Tensors::I2<Dim>(), Tensors::I2<Dim>()) +
               2 * mu * Tensors::I4S<Dim>();
      }

      /**
       * compute the stiffness tensor
       * @param lambda: Lamé's first constant
       * @param mu: Lamé's second constant (i.e., shear modulus)
       */
      inline static T4Mat<Real, Dim> compute_C_T4(const Real & lambda,
                                                  const Real & mu) {
        return lambda * Matrices::Itrac<Dim>() +
               2 * mu * Matrices::Isymm<Dim>();
      }

      /**
       * return stress
       * @param lambda: First Lamé's constant
       * @param mu: Second Lamé's constant (i.e. shear modulus)
       * @param E: Green-Lagrange or small strain tensor
       */
      template <class s_t>
      inline static decltype(auto) evaluate_stress(const Real & lambda,
                                                   const Real & mu, s_t && E) {
        return E.trace() * lambda * Strain_t::Identity() + 2 * mu * E;
      }

      /**
       * return stress and tangent stiffness
       * @param lambda: First Lamé's constant
       * @param mu: Second Lamé's constant (i.e. shear modulus)
       * @param E: Green-Lagrange or small strain tensor
       * @param C: stiffness tensor (Piola-Kirchhoff 2 (or σ) w.r.t to `E`)
       */
      template <class s_t>
      inline static decltype(auto) evaluate_stress(const Real & lambda,
                                                   const Real & mu,
                                                   Tangent_t && C, s_t && E) {
        return std::make_tuple(
            std::move(evaluate_stress(lambda, mu, std::move(E))), std::move(C));
      }
    };

    namespace internal {

      /* ----------------------------------------------------------------------
       */
      template <Dim_t Dim, FiniteDiff FinDif>
      struct NumericalTangentHelper {
        using T4_t = muGrid::T4Mat<Real, Dim>;
        using T2_t = Eigen::Matrix<Real, Dim, Dim>;
        using T2_vec = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

        template <class FunType, class Derived>
        static inline T4_t compute(FunType && fun,
                                   const Eigen::MatrixBase<Derived> & strain,
                                   Real delta);
      };

      /* ----------------------------------------------------------------------
       */
      template <Dim_t Dim, FiniteDiff FinDif>
      template <class FunType, class Derived>
      auto NumericalTangentHelper<Dim, FinDif>::compute(
          FunType && fun, const Eigen::MatrixBase<Derived> & strain, Real delta)
          -> T4_t {
        static_assert((FinDif == FiniteDiff::forward) or
                          (FinDif == FiniteDiff::backward),
                      "Not implemented");
        T4_t tangent{T4_t::Zero()};

        const T2_t fun_val{fun(strain)};
        for (Dim_t i{}; i < Dim * Dim; ++i) {
          T2_t strain2{strain};
          T2_vec strain_vec{strain2.data()};
          switch (FinDif) {
          case FiniteDiff::forward: {
            strain_vec(i) += delta;

            T2_t del_f_del{(fun(strain2) - fun_val) / delta};

            tangent.col(i) = T2_vec(del_f_del.data());
            break;
          }
          case FiniteDiff::backward: {
            strain_vec(i) -= delta;

            T2_t del_f_del{(fun_val - fun(strain2)) / delta};

            tangent.col(i) = T2_vec(del_f_del.data());
            break;
          }
          }
          static_assert(Int(decltype(tangent.col(i))::SizeAtCompileTime) ==
                            Int(T2_t::SizeAtCompileTime),
                        "wrong column size");
        }
        return tangent;
      }

      /* ----------------------------------------------------------------------
       */
      template <Dim_t Dim>
      struct NumericalTangentHelper<Dim, FiniteDiff::centred> {
        using T4_t = muGrid::T4Mat<Real, Dim>;
        using T2_t = Eigen::Matrix<Real, Dim, Dim>;
        using T2_vec = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

        template <class FunType, class Derived>
        static inline T4_t compute(FunType && fun,
                                   const Eigen::MatrixBase<Derived> & strain,
                                   Real delta) {
          T4_t tangent{T4_t::Zero()};

          for (Dim_t i{}; i < Dim * Dim; ++i) {
            T2_t strain1{strain};
            T2_t strain2{strain};
            T2_vec strain1_vec{strain1.data()};
            T2_vec strain2_vec{strain2.data()};
            strain1_vec(i) += delta;
            strain2_vec(i) -= delta;

            T2_t del_f_del{(fun(strain1).eval() - fun(strain2).eval()) /
                           (2 * delta)};

            tangent.col(i) = T2_vec(del_f_del.data());
            static_assert(Int(decltype(tangent.col(i))::SizeAtCompileTime) ==
                              Int(T2_t::SizeAtCompileTime),
                          "wrong column size");
          }
          return tangent;
        }
      };

    }  // namespace internal
    /**
     * Helper function to numerically determine tangent, intended for
     * testing, rather than as a replacement for analytical tangents
     */
    template <Dim_t Dim, FiniteDiff FinDif = FiniteDiff::centred, class FunType,
              class Derived>
    inline muGrid::T4Mat<Real, Dim> compute_numerical_tangent(
        FunType && fun, const Eigen::MatrixBase<Derived> & strain, Real delta) {
      static_assert(Derived::RowsAtCompileTime == Dim,
                    "can't handle dynamic matrix");
      static_assert(Derived::ColsAtCompileTime == Dim,
                    "can't handle dynamic matrix");

      using T2_t = Eigen::Matrix<Real, Dim, Dim>;
      using T2_vec = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

      static_assert(
          std::is_convertible<FunType, std::function<T2_t(T2_t)>>::value,
          "Function argument 'fun' needs to be a function taking "
          "one second-rank tensor as input and returning a "
          "second-rank tensor");

      static_assert(Dim_t(T2_t::SizeAtCompileTime) ==
                        Dim_t(T2_vec::SizeAtCompileTime),
                    "wrong map size");
      return internal::NumericalTangentHelper<Dim, FinDif>::compute(
          std::forward<FunType>(fun), strain, delta);
    }

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIALS_TOOLBOX_HH_
