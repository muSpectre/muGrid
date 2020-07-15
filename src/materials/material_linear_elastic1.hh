/**
 * @file   material_linear_elastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Nov 2017
 *
 * @brief  Implementation for linear elastic reference material like in de Geus
 *         2017. This follows the simplest and likely not most efficient
 *         implementation (with exception of the Python law)
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC1_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC1_HH_

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include <libmugrid/field_map_static.hh>

namespace muSpectre {
  template <Index_t DimM>
  class MaterialLinearElastic1;

  /**
   * traits for objective linear elasticity
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic1<DimM>> {
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
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };

  //! DimM material_dimension (dimension of constitutive law)
  /**
   * implements objective linear elasticity
   */
  template <Index_t DimM>
  class MaterialLinearElastic1
      : public MaterialMuSpectre<MaterialLinearElastic1<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic1, DimM>;

    //! short hand for the type of the elastic tensor
    using Stiffness_t = T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic1>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialLinearElastic1() = delete;

    //! Copy constructor
    MaterialLinearElastic1(const MaterialLinearElastic1 & other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElastic1(const std::string & name,
                           const Index_t & spatial_dimension,
                           const Index_t & nb_quad_pts, const Real & young,
                           const Real & poisson,
                           const std::shared_ptr<muGrid::LocalFieldCollection> &
                               parent_field_collection = nullptr);

    //! Move constructor
    MaterialLinearElastic1(MaterialLinearElastic1 && other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic1() = default;

    //! Copy assignment operator
    MaterialLinearElastic1 &
    operator=(const MaterialLinearElastic1 & other) = delete;

    //! Move assignment operator
    MaterialLinearElastic1 &
    operator=(MaterialLinearElastic1 && other) = delete;

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

    const Stiffness_t & get_C() const;

   protected:
    const Real young;    //!< Young's modulusx
    const Real poisson;  //!< Poisson's ratio
    const Real lambda;   //!< first Lamé constant
    const Real mu;       //!< second Lamé constant (shear modulus)

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
    std::unique_ptr<const Stiffness_t> C_holder;  //!< stiffness tensor
    const Stiffness_t & C;                        //!< ref to stiffness tensor
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialLinearElastic1<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t &
      /*quad_pt_index*/) {
    return Hooke::evaluate_stress(this->lambda, this->mu, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialLinearElastic1<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t &
      /*quad_pt_index*/) {
    using Tangent_t = typename traits::TangentMap_t::reference;
    return Hooke::evaluate_stress(
        this->lambda, this->mu, Tangent_t(const_cast<double *>(this->C.data())),
        E);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC1_HH_
