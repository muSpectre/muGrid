/**
 * @file   material_linear_elastic2.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Feb 2018
 *
 * @brief linear elastic material with imposed eigenstrain and its
 *        type traits. Uses the MaterialMuSpectre facilities to keep it
 *        simple
 *
 * Copyright © 2018 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_

#include "materials/material_linear_elastic1.hh"
#include "common/field.hh"

#include <Eigen/Dense>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM> class MaterialLinearElastic2;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic2<DimS, DimM>> {
    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
        MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! local field_collections used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;
    //! local strain type
    using LStrainMap_t = MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, true>;
    //! elasticity with eigenstrain
    using InternalVariables = std::tuple<LStrainMap_t>;
  };

  /**
   * implements objective linear elasticity with an eigenstrain per pixel
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic2
      : public MaterialMuSpectre<MaterialLinearElastic2<DimS, DimM>, DimS,
                                 DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic2, DimS, DimM>;

    //! type for stiffness tensor construction
    using Stiffness_t =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic2>;

    //! Type of container used for storing eigenstrain
    using InternalVariables = typename traits::InternalVariables;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>;
    //! Default constructor
    MaterialLinearElastic2() = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElastic2(std::string name, Real young, Real poisson);

    //! Copy constructor
    MaterialLinearElastic2(const MaterialLinearElastic2 &other) = delete;

    //! Move constructor
    MaterialLinearElastic2(MaterialLinearElastic2 &&other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic2() = default;

    //! Copy assignment operator
    MaterialLinearElastic2 &
    operator=(const MaterialLinearElastic2 &other) = delete;

    //! Move assignment operator
    MaterialLinearElastic2 &operator=(MaterialLinearElastic2 &&other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class s_t, class eigen_s_t>
    inline decltype(auto) evaluate_stress(s_t &&E, eigen_s_t &&E_eig);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class s_t, class eigen_s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t &&E, eigen_s_t &&E_eig);

    /**
     * return the internals tuple
     */
    InternalVariables &get_internals() { return this->internal_variables; }

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> &pixel) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> &pixel, const StrainTensor &E_eig);

   protected:
    //! linear material without eigenstrain used to compute response
    MaterialLinearElastic1<DimS, DimM> material;
    //! storage for eigenstrain
    using Field_t =
        TensorField<LocalFieldCollection<DimS>, Real, secondOrder, DimM>;
    Field_t &eigen_field;  //!< field holding the eigen strain per pixel
    //! tuple for iterable eigen_field
    InternalVariables internal_variables;

   private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t, class eigen_s_t>
  auto MaterialLinearElastic2<DimS, DimM>::evaluate_stress(s_t &&E,
                                                           eigen_s_t &&E_eig)
      -> decltype(auto) {
    return this->material.evaluate_stress(E - E_eig);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t, class eigen_s_t>
  auto MaterialLinearElastic2<DimS, DimM>::evaluate_stress_tangent(
      s_t &&E, eigen_s_t &&E_eig) -> decltype(auto) {
    // using mat = Eigen::Matrix<Real, DimM, DimM>;
    // mat ecopy{E};
    // mat eig_copy{E_eig};
    // mat ediff{ecopy-eig_copy};
    // std::cout << "eidff - (E-E_eig)" << std::endl << ediff-(E-E_eig) <<
    // std::endl; std::cout << "P1 <internal>" << std::endl <<
    // mat{std::get<0>(this->material.evaluate_stress_tangent(E-E_eig))} <<
    // "</internal>" << std::endl; std::cout << "P2" << std::endl <<
    // mat{std::get<0>(this->material.evaluate_stress_tangent(std::move(ediff)))}
    // << std::endl;
    return this->material.evaluate_stress_tangent(E - E_eig);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_
