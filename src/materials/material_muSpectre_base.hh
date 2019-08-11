/**
 * @file   material_muSpectre_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials written for µSpectre specifically. These
 *         can take full advantage of the configuration-change utilities of
 *         µSpectre. The user can inherit from them to define new constitutive
 *         laws and is merely required to provide the methods for computing the
 *         second Piola-Kirchhoff stress and Tangent. This class uses the
 *         "curiously recurring template parameter" to avoid virtual calls.
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

#ifndef SRC_MATERIALS_MATERIAL_MUSPECTRE_BASE_HH_
#define SRC_MATERIALS_MATERIAL_MUSPECTRE_BASE_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "materials/materials_toolbox.hh"
#include "materials/material_evaluator.hh"
#include "materials/stress_transformations.hh"

#include <libmugrid/field_collection.hh>
#include <libmugrid/field.hh>

#include <tuple>
#include <type_traits>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <utility>

namespace muSpectre {

  // Forward declaration for factory function
  template <Dim_t DimS, Dim_t DimM>
  class CellBase;

  /**
   * material traits are used by `muSpectre::MaterialMuSpectre` to
   * break the circular dependence created by the curiously recurring
   * template parameter. These traits must define
   * - these `muSpectre::FieldMap`s:
   *   - `StrainMap_t`: typically a `muSpectre::MatrixFieldMap` for a
   *                    constant second-order `muSpectre::TensorField`
   *   - `StressMap_t`: typically a `muSpectre::MatrixFieldMap` for a
   *                    writable secord-order `muSpectre::TensorField`
   *   - `TangentMap_t`: typically a `muSpectre::T4MatrixFieldMap` for a
   *                     writable fourth-order `muSpectre::TensorField`
   * - `strain_measure`: the expected strain type (will be replaced by the
   *                     small-strain tensor ε
   *                     `muspectre::StrainMeasure::Infinitesimal` in small
   *                     strain computations)
   * - `stress_measure`: the measure of the returned stress. Is used by
   *                     `muspectre::MaterialMuSpectre` to transform it into
   *                     Cauchy stress (`muspectre::StressMeasure::Cauchy`) in
   *                     small-strain computations and into first
   *                     Piola-Kirchhoff stress `muspectre::StressMeasure::PK1`
   *                     in finite-strain computations
   */

  template <class Material, Dim_t DimS, Dim_t DimM>
  class MaterialMuSpectre;

  /**
   * Base class for most convenient implementation of materials
   */
  template <class Material, Dim_t DimS, Dim_t DimM>
  class MaterialMuSpectre : public MaterialBase<DimS, DimM> {
   public:
    /**
     * type used to determine whether the
     * `muSpectre::MaterialBase::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = MatTB::NeedTangent;
    using Parent = MaterialBase<DimS, DimM>;  //!< base class
    //! global field collection
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    //! expected type for stress fields
    using StressField_t = typename Parent::StressField_t;
    //! expected type for strain fields
    using StrainField_t = typename Parent::StrainField_t;
    //! expected type for tangent stiffness fields
    using TangentField_t = typename Parent::TangentField_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    //! traits for the CRTP subclass
    using traits = MaterialMuSpectre_traits<Material>;
    //  ! expected type to save stress tensor of the pixels of a certian
    // material using
    using MStressField_t = typename Parent::MStressField_t;
    // ! expected type to save stiffness ~ of the pixels of a certian
    // material
    using MTangentField_t = typename Parent::MTangentField_t;
    using Strain_t = typename Parent::Strain_t;
    using Stress_t = typename Parent::Stress_t;
    using Stiffness_t = typename Parent::Stiffness_t;

    //! Default constructor
    MaterialMuSpectre() = delete;

    //! Construct by name
    explicit MaterialMuSpectre(const std::string & name,
                               const Dim_t & spatial_dimension,
                               const Dim_t & nb_quad_pts);

    //! Copy constructor
    MaterialMuSpectre(const MaterialMuSpectre & other) = delete;

    //! Move constructor
    MaterialMuSpectre(MaterialMuSpectre && other) = delete;

    //! Destructor
    virtual ~MaterialMuSpectre() = default;

    //! Factory
    template <class... ConstructorArgs>
    static Material & make(CellBase<DimS, DimM> & cell,
                           ConstructorArgs &&... args);
    /** Factory
     * takes all arguments after the name of the underlying
     * Material's constructor. E.g., if the underlying material is a
     * `muSpectre::MaterialLinearElastic1<threeD>`, these would be Young's
     * modulus and Poisson's ratio.
     */
    template <class... ConstructorArgs>
    static std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>
    make_evaluator(ConstructorArgs &&... args);

    //! Copy assignment operator
    MaterialMuSpectre & operator=(const MaterialMuSpectre & other) = delete;

    //! Move assignment operator
    MaterialMuSpectre & operator=(MaterialMuSpectre && other) = delete;

    // this fucntion is speicalized to assign partial material to a pixel
    template <class... InternalArgs>
    void add_pixel_split(const Ccoord_t<DimS> & pixel, Real ratio,
                         InternalArgs... args);

    void add_pixel_split(const Ccoord_t<DimS> & pixel,
                         Real ratio = 1.0) override;

    // add pixels intersecting to material to the material
    void
    add_split_pixels_precipitate(std::vector<Ccoord_t<DimS>> intersected_pixles,
                                 std::vector<Real> intersection_ratios);

    //! computes stress
    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;
    void compute_stresses(const StrainField_t & F, StressField_t & P,
                          Formulation form,
                          SplitCell is_cell_split = SplitCell::no) final;
    //! computes stress and tangent modulus
    void
    compute_stresses_tangent(const StrainField_t & F, StressField_t & P,
                             TangentField_t & K, Formulation form,
                             SplitCell is_cell_split = SplitCell::no) final;

    inline auto
    constitutive_law_small_strain(const Eigen::Ref<const Strain_t> & strain,
                                  const size_t & pixel_index) -> Stress_t final;

    inline auto
    constitutive_law_finite_strain(const Eigen::Ref<const Strain_t> & strain,
                                   const size_t & pixel_index)
        -> Stress_t final;

    inline auto constitutive_law_tangent_small_strain(
        const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
        -> std::tuple<Stress_t, Stiffness_t> final;

    inline auto constitutive_law_tangent_finite_strain(
        const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
        -> std::tuple<Stress_t, Stiffness_t> final;

   protected:
    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split = SplitCell::no>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split = SplitCell::no>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P, TangentField_t & K)
        __attribute__((visibility("default")));
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  MaterialMuSpectre<Material, DimS, DimM>::MaterialMuSpectre(
      const std::string & name, const Dim_t & spatial_dimension,
      const Dim_t & nb_quad_pts)
    : Parent(name, spatial_dimension, nb_quad_pts) {
    using stress_compatible =
        typename traits::StressMap_t::template is_compatible<StressField_t>;
    using strain_compatible =
        typename traits::StrainMap_t::template is_compatible<StrainField_t>;
    using tangent_compatible =
        typename traits::TangentMap_t::template is_compatible<TangentField_t>;

    static_assert((stress_compatible::value && stress_compatible::explain()),
                  "The material's declared stress map is not compatible "
                  "with the stress field. More info in previously shown "
                  "assert.");

    static_assert((strain_compatible::value && strain_compatible::explain()),
                  "The material's declared strain map is not compatible "
                  "with the strain field. More info in previously shown "
                  "assert.");

    static_assert((tangent_compatible::value && tangent_compatible::explain()),
                  "The material's declared tangent map is not compatible "
                  "with the tangent field. More info in previously shown "
                  "assert.");
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <class... ConstructorArgs>
  Material &
  MaterialMuSpectre<Material, DimS, DimM>::make(CellBase<DimS, DimM> & cell,
                                                ConstructorArgs &&... args) {
    auto mat = std::make_unique<Material>(args...);
    auto & mat_ref = *mat;
    auto is_cell_split{cell.get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell.add_material(std::move(mat));
    return mat_ref;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::add_pixel_split(
      const Ccoord & local_ccoord, Real ratio) {
    auto & this_mat = static_cast<Material &>(*this);
    this_mat.add_pixel(local_ccoord);
    this->assigned_ratio.value().get().push_back(ratio);
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <class... InternalArgs>
  void MaterialMuSpectre<Material, DimS, DimM>::add_pixel_split(
      const Ccoord_t<DimS> & pixel, Real ratio, InternalArgs... args) {
    auto & this_mat = static_cast<Material &>(*this);
    this_mat.add_pixel(pixel, args...);
    this->assigned_ratio.value().get().push_back(ratio);
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::add_split_pixels_precipitate(
      std::vector<Ccoord_t<DimS>> intersected_pixels,
      std::vector<Real> intersection_ratios) {
    // assign precipitate materials:
    for (auto && tup : akantu::zip(intersected_pixels, intersection_ratios)) {
      auto pix = std::get<0>(tup);
      auto ratio = std::get<1>(tup);
      this->add_pixel_split(pix, ratio);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <class... ConstructorArgs>
  std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>
  MaterialMuSpectre<Material, DimS, DimM>::make_evaluator(
      ConstructorArgs &&... args) {
    constexpr Dim_t SpatialDimension{muGrid::Unknown};
    constexpr Dim_t NbQuadPts{1};
    auto mat = std::make_shared<Material>("name", SpatialDimension, NbQuadPts,
                                          args...);
    using Ret_t =
        std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>;
    return Ret_t(mat, MaterialEvaluator<DimM>{mat});
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::compute_stresses(
      const StrainField_t & F, StressField_t & P, Formulation form,
      SplitCell is_cell_split) {
    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::no>(F, P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::simple>(F, P);
        break;
      }
      default:
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::small_strain, SplitCell::no>(
            F, P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::small_strain,
                                      SplitCell::simple>(F, P);
        break;
      }
      default:
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  void MaterialMuSpectre<Material, DimS, DimM>::compute_stresses_tangent(
      const StrainField_t & F, StressField_t & P, TangentField_t & K,
      Formulation form, SplitCell is_cell_split) {
    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::no>(F, P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::simple>(F, P, K);
        break;
      }
      default:
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::small_strain, SplitCell::no>(
            F, P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::small_strain,
                                      SplitCell::simple>(F, P, K);
        break;
      }
      default:
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <Formulation Form, SplitCell is_cell_split>
  void MaterialMuSpectre<Material, DimS, DimM>::compute_stresses_worker(
      const StrainField_t & F, StressField_t & P, TangentField_t & K) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    auto & this_mat = static_cast<Material &>(*this);
    using traits = MaterialMuSpectre_traits<Material>;

    using iterable_proxy_t = typename Parent::template iterable_proxy<
        std::tuple<typename traits::StrainMap_t>,
        std::tuple<typename traits::StressMap_t, typename traits::TangentMap_t>,
        is_cell_split>;

    iterable_proxy_t fields(*this, F, P, K);

    for (auto && arglist : fields) {
      /**
       * arglist is a tuple of three tuples containing only Lvalue
       * references (see value_tye in the class definition of
       * iterable_proxy::iterator). Tuples contain strains, stresses
       * and internal variables, respectively,
       */

      static_assert(std::is_same<typename traits::StrainMap_t::reference,
                                 std::remove_reference_t<decltype(std::get<0>(
                                     std::get<0>(arglist)))>>::value,
                    "Type mismatch for strain reference, check iterator "
                    "value_type");
      static_assert(std::is_same<typename traits::StressMap_t::reference,
                                 std::remove_reference_t<decltype(std::get<0>(
                                     std::get<1>(arglist)))>>::value,
                    "Type mismatch for stress reference, check iterator"
                    "value_type");
      static_assert(std::is_same<typename traits::TangentMap_t::reference,
                                 std::remove_reference_t<decltype(std::get<1>(
                                     std::get<1>(arglist)))>>::value,
                    "Type mismatch for tangent reference, check iterator"
                    "value_type");
      static_assert(
          std::is_same<Real, std::remove_reference_t<decltype(
                                 std::get<3>(arglist))>>::value,
          "Type mismatch for ratio reference, expected a real number");
      auto && strain{std::get<0>(arglist)};
      auto && stress_stiffness{std::get<1>(arglist)};
      auto && stress{std::get<0>(stress_stiffness)};
      auto && stiffness{std::get<1>(stress_stiffness)};
      auto && index{std::get<2>(arglist)};
      auto && ratio{std::get<3>(arglist)};
      if (is_cell_split == SplitCell::simple) {
        auto && stress_stiffness_mat{
            MatTB::constitutive_law_tangent<Form>(this_mat, strain, index)};
        stress += ratio * std::get<0>(stress_stiffness_mat);
        stiffness += ratio * std::get<1>(stress_stiffness_mat);
      } else {
        stress_stiffness =
            MatTB::constitutive_law_tangent<Form>(this_mat, strain, index);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  template <Formulation Form, SplitCell is_cell_split>
  void MaterialMuSpectre<Material, DimS, DimM>::compute_stresses_worker(
      const StrainField_t & F, StressField_t & P) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems
    */
    auto & this_mat = static_cast<Material &>(*this);

    using iterable_proxy_t = typename Parent::template iterable_proxy<
        std::tuple<typename traits::StrainMap_t>,
        std::tuple<typename traits::StressMap_t>, is_cell_split>;

    iterable_proxy_t fields(*this, F, P);

    for (auto && arglist : fields) {
      /**
       * arglist is a tuple of three tuples containing only Lvalue
       * references (see value_type in the class definition of
       * iterable_proxy::iterator). Tuples contain strains, stresses
       * and internal variables, respectively,
       */

      static_assert(std::is_same<typename traits::StrainMap_t::reference,
                                 std::remove_reference_t<decltype(std::get<0>(
                                     std::get<0>(arglist)))>>::value,
                    "Type mismatch for strain reference, check iterator "
                    "value_type");
      static_assert(std::is_same<typename traits::StressMap_t::reference,
                                 std::remove_reference_t<decltype(std::get<0>(
                                     std::get<1>(arglist)))>>::value,
                    "Type mismatch for stress reference, check iterator"
                    "value_type");
      static_assert(
          std::is_same<Real, std::remove_reference_t<decltype(
                                 std::get<3>(arglist))>>::value,
          "Type mismatch for ratio reference, expected a real number");

      auto && strain{std::get<0>(arglist)};
      auto && stress{std::get<0>(std::get<1>(arglist))};
      auto && index{std::get<2>(arglist)};
      auto && ratio{std::get<3>(arglist)};

      if (is_cell_split == SplitCell::simple) {
        stress +=
            ratio * MatTB::constitutive_law<Form>(this_mat, strain, index);
      } else {
        stress = MatTB::constitutive_law<Form>(this_mat, strain, index);
      }
    }
  }
  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  auto MaterialMuSpectre<Material, DimS, DimM>::constitutive_law_finite_strain(
      const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
      -> Stress_t {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());
    return std::move(MatTB::constitutive_law<Formulation::finite_strain>(
        this_mat, std::make_tuple(F), pixel_index));
  }
  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  auto MaterialMuSpectre<Material, DimS, DimM>::constitutive_law_small_strain(
      const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
      -> Stress_t {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());
    return std::move(MatTB::constitutive_law<Formulation::small_strain>(
        this_mat, std::make_tuple(F), pixel_index));
  }
  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  auto MaterialMuSpectre<Material, DimS, DimM>::
      constitutive_law_tangent_finite_strain(
          const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
          -> std::tuple<Stress_t, Stiffness_t> {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());
    return std::move(
        MatTB::constitutive_law_tangent<Formulation::finite_strain>(
            this_mat, std::make_tuple(F), pixel_index));
  }
  /* ---------------------------------------------------------------------- */
  template <class Material, Dim_t DimS, Dim_t DimM>
  auto MaterialMuSpectre<Material, DimS, DimM>::
      constitutive_law_tangent_small_strain(
          const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
          -> std::tuple<Stress_t, Stiffness_t> {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());
    return std::move(MatTB::constitutive_law_tangent<Formulation::small_strain>(
        this_mat, std::make_tuple(F), pixel_index));
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_MUSPECTRE_BASE_HH_
