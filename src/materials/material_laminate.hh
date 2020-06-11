/**
 * @file   material_laminate.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   18 May 2020
 *
 * @brief  the material that uses the laminate homogenisation for a single pixel
 * stress and tangent evaluation. This material takes shared_ptrs to materials
 * and for each pixel it expects an assignment ratio for its constituent
 * materials as well as the normal vector of the materials interface plane and
 * takes them as two laminate layers of those materials touching at the
 * interfacial plane direction.
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

#ifndef SRC_MATERIALS_MATERIAL_LAMINATE_HH_
#define SRC_MATERIALS_MATERIAL_LAMINATE_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_muSpectre_base.hh"

#include "cell/cell.hh"

#include "libmugrid/T4_map_proxy.hh"

#include <vector>

namespace muSpectre {
  template <Index_t DimM, Formulation Form>
  class MaterialLaminate;

  template <Index_t DimM>
  struct MaterialMuSpectre_traits<
      MaterialLaminate<DimM, Formulation::finite_strain>> {
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
    constexpr static auto stress_measure{StressMeasure::PK1};
  };

  template <Index_t DimM>
  struct MaterialMuSpectre_traits<
      MaterialLaminate<DimM, Formulation::small_strain>> {
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

  template <Index_t DimM, Formulation Form>
  class MaterialLaminate
      : public MaterialMuSpectre<MaterialLaminate<DimM, Form>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLaminate<DimM, Form>, DimM>;
    //
    using MatPtr_t = std::shared_ptr<MaterialBase>;

    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    using MappedVectorField_t =
        muGrid::MappedT1Field<Real, Mapping::Mut, DimM, IterUnit::SubPt>;
    using MappedScalarField_t =
        muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt>;

    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffness
     */
    using NeedTangent = MatTB::NeedTangent;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLaminate<DimM, Form>>;

    //! Default constructor
    MaterialLaminate() = delete;

    //! Constructor with name and material properties
    MaterialLaminate(
        const std::string & name, const Index_t & spatial_dimension,
        const Index_t & nb_quad_pts,
        std::shared_ptr<muGrid::LocalFieldCollection> parent_field = nullptr);

    //! Copy constructor
    MaterialLaminate(const MaterialLaminate & other) = delete;

    //! Move constructor
    MaterialLaminate(MaterialLaminate && other) = delete;

    //! Destructor
    virtual ~MaterialLaminate() = default;

    /**
     * evaluates first Piola-Kirchhoff stress given the Gradient
     */

    template <typename Derived>
    T2_t evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                         const size_t & pixel_index);

    /**
     * evaluates first Piola-Kirchhoff stress and its corresponding tangent
     * given the Gradient
     */
    template <typename Derived>
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & pixel_index);

    /**
     * override add_pixel
     */
    void add_pixel(const size_t & pixel_id) final;

    /**
     * overload add_pixel to add underlying materials and their ratio and
     * interface direction to the material laminate
     */
    void add_pixel(
        const size_t & pixel_id, MatPtr_t mat1, MatPtr_t mat2,
        const Real & ratio,
        const Eigen::Ref<const Eigen::Matrix<Real, DimM, 1>> & normal_Vector);

    /**
     * This function adds pixels according to the precipitate intersected pixels
     * and the materials involved
     */
    void add_pixels_precipitate(
        const std::vector<Ccoord_t<DimM>> & intersected_pixels,
        const std::vector<Index_t> & intersected_pixels_id,
        const std::vector<Real> & intersection_ratios,
        const std::vector<Eigen::Matrix<Real, DimM, 1>> & intersection_normals,
        MatPtr_t mat1, MatPtr_t mat2);

   protected:
    MappedVectorField_t
        normal_vector_field;  //!< field holding the normal vector
                              //!< of the interface of the layers

    MappedScalarField_t
        volume_ratio_field;  //!< field holding the normal vector

    std::vector<MatPtr_t>
        material_left_vector{};  //!< "left" material contained in a laminate
    std::vector<MatPtr_t>
        material_right_vector{};  //!< "right" material contained in a laminate
  };

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LAMINATE_HH_
