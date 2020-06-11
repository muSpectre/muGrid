/**
 * @file   material_neo_hookean_elastic.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   27 Feb 2020
 *
 * @brief  The Neo-Hookean material (Adapted from: Simo JC, Hughes TJ.
 * Computational inelasticity. Springer Science & Business Media; 2006 May 7)
 *
 * Copyright © 2020 Ali Falsafi
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

#ifndef SRC_MATERIALS_MATERIAL_NEO_HOOKEAN_ELASTIC_HH_
#define SRC_MATERIALS_MATERIAL_NEO_HOOKEAN_ELASTIC_HH_

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations_Kirchhoff.hh"

#include <iostream>

namespace muSpectre {
  template <Index_t DimM>
  class MaterialNeoHookeanElastic;

  /**
   * traits for objective linear Neo-Hookean material
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialNeoHookeanElastic<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};
  };

  template <Index_t DimM>
  class MaterialNeoHookeanElastic
    : public MaterialMuSpectre<MaterialNeoHookeanElastic<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialNeoHookeanElastic<DimM>, DimM>;

    //! short hand for the type of the elastic tensor
    using Stiffness_t = T4Mat<Real, DimM>;
    using Stress_t = Eigen::Matrix<Real, DimM, DimM>;
    using Strain_t = Stress_t;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialNeoHookeanElastic>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialNeoHookeanElastic() = delete;

    //! Copy constructor
    MaterialNeoHookeanElastic(const MaterialNeoHookeanElastic & other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialNeoHookeanElastic(const std::string & name,
                              const Index_t & spatial_dimension,
                              const Index_t & nb_quad_pts, const Real & young,
                              const Real & poisson);

    //! Move constructor
    MaterialNeoHookeanElastic(MaterialNeoHookeanElastic && other) = delete;

    //! Destructor
    virtual ~MaterialNeoHookeanElastic() = default;

    //! Copy assignment operator
    MaterialNeoHookeanElastic &
    operator=(const MaterialNeoHookeanElastic & other) = delete;

    //! Move assignment operator
    MaterialNeoHookeanElastic &
    operator=(MaterialNeoHookeanElastic && other) = delete;

    /**
     * calculation of volumetric part of the stress
     */
    inline Real evaluate_elastic_volumetric_stress(const Real & J);

    /**
     * calculation of deviatoric part of the stress
     */
    inline Stress_t evaluate_elastic_deviatoric_stress(
        const Eigen::Ref<const Stress_t> & E_dev);
    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & /*quad_pt_index*/);
    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & /*quad_pt_index*/);

   protected:
    const Real young;    //!< Young's modulusx
    const Real poisson;  //!< Poisson's ratio
    const Real lambda;   //!< first Lamé constant
    const Real mu;       //!< second Lamé constant (shear modulus)
    const Real K;        //!< Bulk Modulus

    // Here, the stiffness tensor is encapsulated into a unique ptr because of
    // this bug:
    // https://eigen.tuxfamily.narkive.com/maHiFSha/fixed-size-vectorizable-members-and-std-make-shared
    // . The problem is that `std::make_shared` uses the global `::new` to
    // allocate `void *` rather than using the the object's `new` operator, and
    // therefore ignores the solution proposed by eigen (documented here
    // http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html).
    // Offloading the offending object into a heap-allocated structure who's
    // construction we control fixes this problem temporarily, until we can use
    // C++17 and guarantee alignment. This comes at the cost of a heap
    // allocation, which is not an issue here, as this happens only once per
    // material and run.
    std::unique_ptr<const Stiffness_t> C_linear_holder;  //!< stiffness tensor
    const Stiffness_t & C_linear;  //!< ref to stiffness tensor
  };
  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialNeoHookeanElastic<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & F, const size_t &
      /*quad_pt_index*/) -> decltype(auto) {
    auto && J{F.determinant()};  //! Volumetric part of the gradient
    auto && F_dev{std::pow(J, -(1.0 / 3.0)) * F};
    auto && RC_dev{F_dev.transpose() * F_dev};
    auto && E_dev = 0.5 * (RC_dev - Stress_t::Identity());
    auto && JP{this->evaluate_elastic_volumetric_stress(J) *
               Stress_t::Identity()};

    // I could not avoid explicit construction of the following variable(s):
    // they result in wrong answer(memory bug) if I choose auto && for their
    // types
    Stress_t tau_dev{F_dev * this->evaluate_elastic_deviatoric_stress(E_dev) *
                     F_dev.transpose()};
    Stress_t tau{tau_dev + JP};
    return tau;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialNeoHookeanElastic<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & F, const size_t &
      /*quad_pt_index*/) -> decltype(auto) {
    auto && J{F.determinant()};  //!< Volumetric part of the gradient
    auto && F_dev{std::pow(J, -(1.0 / 3.0)) * F};

    auto && RC_dev{F_dev.transpose() * F_dev};
    auto && E_dev = 0.5 * (RC_dev - Stress_t::Identity());

    // Compute initial elastic stress Kirchhoff tensor
    auto && JP{this->evaluate_elastic_volumetric_stress(J) *
               Stress_t::Identity()};

    // I could not avoid explicit construction of the following variable(s):
    // they result in wrong answer(memory bug) if I choose auto && for their
    // types
    Stress_t tau_dev{F_dev * this->evaluate_elastic_deviatoric_stress(E_dev) *
                     F_dev.transpose()};
    Stress_t tau{MatTB::compute_deviatoric<DimM>(tau_dev) + JP};
    auto && tau_bar{MatTB::compute_deviatoric<DimM>(tau)};

    // I could not avoid explicit construction of the following variable(s):
    // they result in wrong answer(memory bug) if I choose auto && for their
    // types
    Stiffness_t c_bar{
        muGrid::Matrices::AxisTransform::push_forward(this->C_linear, F)};
    Stiffness_t c{
        c_bar - ((2.0 / 3.0) * (Matrices::outer(tau_bar, Stress_t::Identity()) +
                                Matrices::outer(Stress_t::Identity(), tau_bar) -
                                (tau.trace() * Matrices::Iasymm<DimM>())))};

    Stiffness_t dtau_dF{Stiffness_t::Zero()};
    // Conversion of ∂τ/∂E to ∂τ/∂F(desired return measure for the material)
    for (int i{0}; i < DimM; ++i) {
      for (int j{0}; j < DimM; ++j) {
        for (int k{0}; k < DimM; ++k) {
          for (int l{0}; l < DimM; ++l) {
            for (int m{0}; m < DimM; ++m) {
              get(dtau_dF, i, j, k, l) += get(c, i, j, m, l) * F(k, m);
            }
          }
        }
      }
    }

    return std::make_tuple(tau, dtau_dF);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialNeoHookeanElastic<DimM>::evaluate_elastic_volumetric_stress(
      const Real & J) {
    // U(Θ) = [K/4]* [(Θ-1)² + ln(Θ)²] ⇒
    Real && K{(DimM * this->lambda + 2 * this->mu) / DimM};
    // Jp :
    auto && stress{0.5 * K * (J * J - J + std::log(J))};
    return stress;
    // return 0.5 * K * (J * J - J + std::log(J));
  }

  /* ----------------------------------------------------------------------
   */
  template <Index_t DimM>
  auto MaterialNeoHookeanElastic<DimM>::evaluate_elastic_deviatoric_stress(
      const Eigen::Ref<const Stress_t> & E_dev) -> Stress_t {
    return (2 * this->mu * MatTB::compute_deviatoric<DimM>(E_dev));
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_NEO_HOOKEAN_ELASTIC_HH_
