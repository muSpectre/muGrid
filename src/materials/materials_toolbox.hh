/**
 * @file   materials_toolbox.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
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

#ifndef SRC_MATERIALS_MATERIALS_TOOLBOX_HH_
#define SRC_MATERIALS_MATERIALS_TOOLBOX_HH_

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_PK1.hh"

#include "common/voigt_conversion.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/exception.hh>
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
  template <class Material>
  struct MaterialMuSpectre_traits {};

  namespace MatTB {
    /**
     * thrown when generic materials-related runtime errors occur
     * (mostly continuum mechanics problems)
     */
    class MaterialsToolboxError : public muGrid::RuntimeError {
     public:
      //! constructor
      explicit MaterialsToolboxError(const std::string & what)
          : muGrid::RuntimeError(what) {}
      //! constructor
      explicit MaterialsToolboxError(const char * what)
          : muGrid::RuntimeError(what) {}
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

    /* ---------------------------------------------------------------------- */
    namespace internal {

      /** Structure for functions returning one strain measure as a
       *  function of another
       **/
      template <StrainMeasure In, StrainMeasure Out = In>
      struct ConvertStrain {
        // static_assert((In == StrainMeasure::Gradient) ||
        //                   (In == StrainMeasure::Infinitesimal),
        //               "This situation makes me suspect that you are not using
        //               " "MatTb as intended. Disable this assert only if you
        //               are " "sure about what you are doing.");

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
      inline static auto evaluate_stress(const Real & lambda, const Real & mu,
                                         s_t && E) -> decltype(auto) {
        return E.trace() * lambda * Strain_t::Identity() + 2 * mu * E;
      }

      /**
       * return stress
       * @param C: stiffness tensor (Piola-Kirchhoff 2 (or σ) w.r.t to `E`)
       * @param E: Green-Lagrange or small strain tensor
       */
      template <class T_t, class s_t>
      inline static auto evaluate_stress(const T_t C, s_t && E)
          -> decltype(auto) {
        return Matrices::tensmult(C, E);
      }

      /**
       * return stress and tangent stiffness
       * @param lambda: First Lamé's constant
       * @param mu: Second Lamé's constant (i.e. shear modulus)
       * @param E: Green-Lagrange or small strain tensor
       * @param C: stiffness tensor (Piola-Kirchhoff 2 (or σ) w.r.t to `E`)
       */
      template <class s_t>
      inline static auto evaluate_stress(const Real & lambda, const Real & mu,
                                         Tangent_t && C, s_t && E)
          -> decltype(auto) {
        return std::make_tuple(
            std::move(evaluate_stress(lambda, mu, std::move(E))), std::move(C));
      }
    };

    namespace internal {

      /**
       * implementation-structure for computing numerical tangents. For internal
       * use only.
       * @tparam Dim dimensionality of the material
       * @tparam FinDif specificaition of the type of finite differences
       */
      template <Dim_t Dim, FiniteDiff FinDif>
      struct NumericalTangentHelper {
        //! short-hand for fourth-rank tensors
        using T4_t = muGrid::T4Mat<Real, Dim>;
        //! short-hand for second-rank tensors
        using T2_t = Eigen::Matrix<Real, Dim, Dim>;
        //! short-hand for second-rank tensor reshaped to a vector
        using T2_vec = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

        //! compute and return the approximate tangent moduli at strain `strain`
        template <class FunType, class Derived>
        static inline T4_t compute(FunType && fun,
                                   const Eigen::MatrixBase<Derived> & strain,
                                   Real delta) {
          static_assert((FinDif == FiniteDiff::forward) or
                            (FinDif == FiniteDiff::backward),
                        "should use specialised version");
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
      };

      /**
       * specialisation for centred differences
       */
      template <Dim_t Dim>
      struct NumericalTangentHelper<Dim, FiniteDiff::centred> {
        //! short-hand for fourth-rank tensors
        using T4_t = muGrid::T4Mat<Real, Dim>;
        //! short-hand for second-rank tensors
        using T2_t = Eigen::Matrix<Real, Dim, Dim>;
        //! short-hand for second-rank tensor reshaped to a vector
        using T2_vec = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

        //! compute and return the approximate tangent moduli at strain `strain`
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

      static_assert(Index_t(T2_t::SizeAtCompileTime) ==
                        Index_t(T2_vec::SizeAtCompileTime),
                    "wrong map size");
      return internal::NumericalTangentHelper<Dim, FinDif>::compute(
          std::forward<FunType>(fun), strain, delta);
    }

    /**
     * Computes the deviatoric stress σ_{dev}=σ-\frac{1}{3} tr(σ)*I, on each
     * pixel from a given stress, first only for PK2.
     */
    template <Dim_t DimM>
    inline Eigen::Matrix<Real, DimM, DimM>
    compute_deviatoric(const Eigen::Matrix<Real, DimM, DimM> & matrix) {
      //! compute deviatoric stress tensor σ^{dev}=σ-\frac{1}{dim} tr(σ) I
      return matrix -
             (1. / DimM) *
                 (matrix.trace() * Eigen::Matrix<Real, DimM, DimM>::Identity());
    }

    /**
     * Computes the equivalent von Mises stress σ_{eq} on each pixel from a
     * given PK2 stress.
     */
    template <Dim_t DimM>
    inline decltype(auto) compute_equivalent_von_Mises_stress(
        const Eigen::Map<const Eigen::Matrix<Real, DimM, DimM>> PK2) {
      Eigen::Matrix<Real, DimM, DimM> PK2_matrix = PK2;
      if (DimM == 3) {
        auto && deviatoric_stress = compute_deviatoric<DimM>(PK2_matrix);
        //! 3D case:
        //! compute σ_{eq} = \sqrt{\frac{3}{2} σ^{dev} : σ^{dev}}
        //!                = \sqrt{\frac{3}{2} tr(σᵈᵉᵛ·(σᵈᵉᵛ)ᵀ)}
        const Real equivalent_stress{
            sqrt(3. / 2. *
                 (deviatoric_stress * deviatoric_stress.transpose()).trace())};
        return equivalent_stress;
      } else if (DimM == 2) {
        //! 2D case:
        //! For the 2D von Mises stress we assume a general plane stress
        //! (σ₃₃=σ₃₁=σ₃₂=0) state.
        //! σ_{eq} = \sqrt{σ₁₁² + σ₂₂² - σ₁₁σ₂₂ + 3σ₁₂²}
        //! Bruchmechanik 6. edition (2016); Dietmar Gross, Thomas Seelig;
        //! DOI 10.1007/978-3-662-46737-4; chap. 1.3.3.1, eq(1.78)
        const Real equivalent_stress{
            sqrt(PK2(0, 0) * PK2(0, 0) + PK2(1, 1) * PK2(1, 1) -
                 PK2(0, 0) * PK2(1, 1) + 3 * PK2(0, 1) * PK2(0, 1))};
        return equivalent_stress;
      } else if (DimM == 0) {
        //! 1D case:
        const Real equivalent_stress{PK2(0, 0)};
        return equivalent_stress;
      }
    }

    /* ----------------------------------------------------------------------*/
    struct OperationAddition {
      explicit OperationAddition(const Real & ratio) : ratio{ratio} {};
      const Real & ratio;

      template <typename Derived1, typename Derived2>
      void operator()(const Eigen::MatrixBase<Derived1> & material_stress,
                      Eigen::MatrixBase<Derived2> & stored_stress) const {
        stored_stress += this->ratio * material_stress;
      }
    };

    /* ----------------------------------------------------------------------*/
    struct OperationAssignment {
      template <typename Derived1, typename Derived2>
      void operator()(const Eigen::MatrixBase<Derived1> & material_stress,
                      Eigen::MatrixBase<Derived2> & stored_stress) const {
        stored_stress = material_stress;
      }
    };

    /* ---------------------------------------------------------------------- */
    template <StoreNativeStress StoreNative = StoreNativeStress::no,
              Dim_t Dim = 1>  // arbitrary number, only here for specialisation>
    struct NativeStressTreatment {
      template <typename Derived>
      void
      operator()(const Eigen::MatrixBase<Derived> & /*native_stress*/) const {
        // do absolutely nothing by default
      }
    };

    /* ---------------------------------------------------------------------- */
    template <Dim_t Dim>
    struct NativeStressTreatment<StoreNativeStress::yes, Dim> {
      explicit NativeStressTreatment(
          Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress)
          : native_stress_storage{native_stress} {}

      template <typename Derived>
      void operator()(const Eigen::MatrixBase<Derived> & native_stress) {
        this->native_stress_storage = native_stress;
      }

     protected:
      Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> native_stress_storage;
    };

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strain, class Stress,
              class Op, class NativeTreat>
    void evaluate_material_stress(Material & mat, Strain && strain,
                                  Stress & stress, const size_t & quad_pt_id,
                                  const Op & operation,
                                  NativeTreat & native_stress_treatment) {
      using traits = MaterialMuSpectre_traits<Material>;

      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
          get_formulation_strain_type(Form, traits::strain_measure)};

      switch (Form) {
      case Formulation::small_strain: {
        auto && eps{MatTB::convert_strain<stored_strain_m, expected_strain_m>(
            std::get<0>(strain))};

        auto && stress_result{mat.evaluate_stress(eps, quad_pt_id)};
        // the following is a no-op if store_native_stress in not 'yes'
        native_stress_treatment(stress_result);

        // stress evaluation:
        operation(stress_result, stress);
        break;
      }
      case Formulation::finite_strain: {
        auto && grad{std::get<0>(strain)};
        auto && E{
            MatTB::convert_strain<stored_strain_m, expected_strain_m>(grad)};
        auto && stress_result{mat.evaluate_stress(std::move(E), quad_pt_id)};

        // the following is a no-op if store_native_stress in not 'yes'
        native_stress_treatment(stress_result);

        operation(::muSpectre::MatTB::PK1_stress<traits::stress_measure,
                                                 traits::strain_measure>(
                      std::move(grad), std::move(stress_result)),
                  stress);
        break;
      }
      case Formulation::native: {
        auto && strain_converted{
            MatTB::convert_strain<stored_strain_m, expected_strain_m>(
                std::get<0>(strain))};

        operation(mat.evaluate_stress(std::move(strain_converted), quad_pt_id),
                  stress);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown formulation");
        break;
      }
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses>
    void constitutive_law(Material & mat, Strains && strains,
                          Stresses & stresses, const size_t & quad_pt_id,
                          const Real & ratio) {
      OperationAddition operation_addition(ratio);
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress<Form>(mat, strains, stresses, quad_pt_id,
                                     operation_addition, stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses>
    void constitutive_law(Material & mat, Strains && strains,
                          Stresses & stresses, const size_t & quad_pt_id) {
      OperationAssignment operation_assignment;
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress<Form>(mat, strains, stresses, quad_pt_id,
                                     operation_assignment, stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses,
              Dim_t Dim = Material::MaterialDimension()>
    void constitutive_law(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id, const Real & ratio,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      OperationAddition operation_addition(ratio);
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress<Form>(mat, strains, stresses, quad_pt_id,
                                     operation_addition, stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses,
              Dim_t Dim = Material::MaterialDimension()>
    void constitutive_law(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      OperationAssignment operation_assignment;
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress<Form>(mat, strains, stresses, quad_pt_id,
                                     operation_assignment, stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strain, class Stress,
              class Stiffness, class Op, class NativeTreat>
    void evaluate_material_stress_tangent(
        Material & mat, Strain && strain,
        std::tuple<Stress, Stiffness> & stress_stiffness,
        const size_t & quad_pt_id, const Op & operation,
        NativeTreat & native_stress_treatment) {
      using traits = MaterialMuSpectre_traits<Material>;

      constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
      constexpr StrainMeasure expected_strain_m{
          get_formulation_strain_type(Form, traits::strain_measure)};
      switch (Form) {
      case Formulation::small_strain: {
        auto && eps{MatTB::convert_strain<stored_strain_m, expected_strain_m>(
            std::get<0>(strain))};

        auto && stress_stiffness_mat{
            mat.evaluate_stress_tangent(std::move(eps), quad_pt_id)};
        // the following is a no-op if store_native_stress in not 'yes'
        native_stress_treatment(std::get<0>(stress_stiffness_mat));
        operation(std::get<0>(stress_stiffness_mat),
                  std::get<0>(stress_stiffness));
        operation(std::get<1>(stress_stiffness_mat),
                  std::get<1>(stress_stiffness));
        break;
      }
      case Formulation::finite_strain: {
        auto && grad{std::get<0>(strain)};
        auto E{MatTB::convert_strain<stored_strain_m, expected_strain_m>(grad)};
        auto && stress_stiffness_mat{
            mat.evaluate_stress_tangent(std::move(E), quad_pt_id)};

        // the following is a no-op if store_native_stress in not 'yes'
        native_stress_treatment(std::get<0>(stress_stiffness_mat));
        auto && stress_stiffness_mat_converted{
            ::muSpectre::MatTB::PK1_stress<traits::stress_measure,
                                           traits::strain_measure>(
                std::move(grad), std::move(std::get<0>(stress_stiffness_mat)),
                std::move(std::get<1>(stress_stiffness_mat)))};

        operation(std::get<0>(stress_stiffness_mat_converted),
                  std::get<0>(stress_stiffness));
        operation(std::get<1>(stress_stiffness_mat_converted),
                  std::get<1>(stress_stiffness));
        break;
      }
      case Formulation::native: {
        auto && strain_converted{
            MatTB::convert_strain<stored_strain_m, expected_strain_m>(
                std::get<0>(strain))};
        auto && stress_stiffness_mat{mat.evaluate_stress_tangent(
            std::move(strain_converted), quad_pt_id)};
        native_stress_treatment(std::get<0>(stress_stiffness_mat));

        operation(std::get<0>(stress_stiffness_mat),
                  std::get<0>(stress_stiffness));
        operation(std::get<1>(stress_stiffness_mat),
                  std::get<1>(stress_stiffness));
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown formualtion");
        break;
      }
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses>
    void constitutive_law_tangent(Material & mat, Strains && strains,
                                  Stresses & stresses,
                                  const size_t & quad_pt_id) {
      OperationAssignment operation_assignment{};
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress_tangent<Form>(mat, strains, stresses, quad_pt_id,
                                             operation_assignment,
                                             stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses,
              Dim_t Dim = Material::MaterialDimension()>
    void constitutive_law_tangent(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      static_assert(Dim == Material::MaterialDimension(),
                    "Dim needs to be a deduced parameter, do not touch it");
      OperationAssignment operation_assignment{};
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress_tangent<Form>(mat, strains, stresses, quad_pt_id,
                                             operation_assignment,
                                             stress_treatment);
    }

    /*----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses>
    void constitutive_law_tangent(Material & mat, Strains && strains,
                                  Stresses & stresses,
                                  const size_t & quad_pt_id,
                                  const Real & ratio) {
      OperationAddition operation_addition{ratio};
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress_tangent<Form>(mat, strains, stresses, quad_pt_id,
                                             operation_addition,
                                             stress_treatment);
    }

    /*----------------------------------------------------------------------*/
    template <Formulation Form, class Material, class Strains, class Stresses,
              Dim_t Dim = Material::MaterialDimension()>
    void constitutive_law_tangent(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id, const Real & ratio,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      static_assert(Dim == Material::MaterialDimension(),
                    "Dim needs to be a deduced parameter, do not touch it");
      OperationAddition operation_addition{ratio};
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress_tangent<Form>(mat, strains, stresses, quad_pt_id,
                                             operation_addition,
                                             stress_treatment);
    }

    /*----------------------------------------------------------------------*/
    template <Dim_t DimM, class Derived1, class Derived2>
    void make_C_from_C_voigt(const Eigen::MatrixBase<Derived1> & C_voigt,
                             Eigen::MatrixBase<Derived2> & C_holder) {
      using muGrid::get;
      using VC_t = VoigtConversion<DimM>;
      constexpr Dim_t VSize{vsize(DimM)};
      if (not(C_voigt.rows() == VSize) or not(C_voigt.cols() == VSize)) {
        std::stringstream err_str{};
        err_str << "The stiffness tensor should be input as a " << VSize
                << " × " << VSize << " Matrix in Voigt notation. You supplied"
                << " a " << C_voigt.rows() << " × " << C_voigt.cols()
                << " matrix" << std::endl;
        throw(muGrid::RuntimeError(err_str.str()));
      }

      const auto & sym_mat{VC_t::get_sym_mat()};
      for (int i{0}; i < DimM; ++i) {
        for (int j{0}; j < DimM; ++j) {
          for (int k{0}; k < DimM; ++k) {
            for (int l{0}; l < DimM; ++l) {
              get(C_holder, i, j, k, l) = C_voigt(sym_mat(i, j), sym_mat(k, l));
            }
          }
        }
      }
    }

    /*----------------------------------------------------------------------*/
  }  // namespace MatTB
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIALS_TOOLBOX_HH_
