/**
 * @file   material_muSpectre.hh
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

#ifndef SRC_MATERIALS_MATERIAL_MUSPECTRE_HH_
#define SRC_MATERIALS_MATERIAL_MUSPECTRE_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_mechanics_base.hh"
#include "materials/material_base.hh"
#include "materials/materials_toolbox.hh"
#include "materials/material_evaluator.hh"
#include "materials/iterable_proxy.hh"

#include "cell/cell.hh"
#include "cell/cell_data.hh"

#include <libmugrid/field_map_static.hh>
#include <libmugrid/physics_domain.hh>

#include <tuple>
#include <type_traits>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <utility>

#include "sstream"

namespace muSpectre {

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
  template <class Material>
  struct MaterialMuSpectre_traits {};

  template <class Material, Index_t DimM, class Parent>
  class MaterialMuSpectre;


  /**
   * default traits class for scaral problems, should work for heat flux or
   * electric field problems
   */
  template <Index_t DimM>
  struct DefaultScalar_traits {
    //! expected map type for gradient fields
    using StrainMap_t =
        muGrid::T1FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for flux fields
    using StressMap_t =
        muGrid::T1FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;
    //! expected map type for tangent fields
    using TangentMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    // plain type for gradient representation (termed "strain" from mechanics)
    using Strain_t = Eigen::Matrix<Real, DimM, 1>;
    // plain type for flux representation (termed "stress" from mechanics)
    using Stress_t = Strain_t;
    // plain type for tangent representation
    using Tangent_t = Eigen::Matrix<Real, DimM, DimM>;
  };

  /**
   * Base class for most convenient implementation of materials
   */
  template <class Material, Index_t DimM, class Parent_ = MaterialBase>
  class MaterialMuSpectre : public Parent_ {
   public:
    using Parent = Parent_;  //!< base class

    //! traits for the CRTP subclass
    using traits = MaterialMuSpectre_traits<Material>;
    using DynMatrix_t = typename Parent::DynMatrix_t;

    using Strain_t = typename traits::Strain_t;
    using Stress_t = typename traits::Stress_t;
    using Tangent_t = typename traits::Tangent_t;

    //! Default constructor
    MaterialMuSpectre() = delete;

    //! Construct by name
    MaterialMuSpectre(const std::string & name,
                      const Index_t & spatial_dimension,
                      const Index_t & nb_quad_pts,
                      const std::shared_ptr<muGrid::LocalFieldCollection> &
                          parent_field_collection = nullptr);

    //! Copy constructor
    MaterialMuSpectre(const MaterialMuSpectre & other) = delete;

    //! Move constructor
    MaterialMuSpectre(MaterialMuSpectre && other) = delete;

    //! Destructor
    virtual ~MaterialMuSpectre() = default;

    //! Factory. The ConstructorArgs refer the arguments after `name`
    template <class... ConstructorArgs>
    inline static Material & make(std::shared_ptr<Cell> cell,
                                  const std::string & name,
                                  ConstructorArgs &&... args);
    //! Factory. The ConstructorArgs refer the arguments after `name`
    template <class... ConstructorArgs>
    inline static Material & make(std::shared_ptr<CellData> cell,
                                  const std::string & name,
                                  ConstructorArgs &&... args);
    /** Factory
     * takes all arguments after the name of the underlying
     * Material's constructor. E.g., if the underlying material is a
     * `muSpectre::MaterialLinearElastic1<threeD>`, these would be Young's
     * modulus and Poisson's ratio.
     */
    template <class... ConstructorArgs>
    inline static std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>
    make_evaluator(ConstructorArgs &&... args);

    //! Copy assignment operator
    MaterialMuSpectre & operator=(const MaterialMuSpectre & other) = delete;

    //! Move assignment operator
    MaterialMuSpectre & operator=(MaterialMuSpectre && other) = delete;

    // this fucntion is speicalized to assign partial material to a pixel
    template <class... InternalArgs>
    inline void add_pixel_split(const size_t & pixel_id, Real ratio,
                                InternalArgs... args);

    // add pixels intersecting to material to the material
    inline void add_split_pixels_precipitate(
        const std::vector<size_t> & intersected_pixel_ids,
        const std::vector<Real> & intersection_ratios);

    //! computes stress
    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;

    //! computes stress
    inline void
    compute_stresses(const muGrid::RealField & F, muGrid::RealField & P,
                     const SplitCell & is_cell_split = SplitCell::no,
                     const StoreNativeStress & store_native_stress =
                         StoreNativeStress::no) override;

    //! computes stress and tangent modulus
    inline void
    compute_stresses_tangent(const muGrid::RealField & F, muGrid::RealField & P,
                             muGrid::RealField & K,
                             const SplitCell & is_cell_split = SplitCell::no,
                             const StoreNativeStress & store_native_stress =
                                 StoreNativeStress::no) override;

    //! return the material dimension at compile time
    constexpr static Index_t MaterialDimension() { return DimM; }

    inline std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & pixel_index) override;

    //! returns whether or not a field with native stress has been stored
    inline bool has_native_stress() const final;

    /**
     * returns the stored native stress field. Throws a runtime error if native
     * stress has not been stored
     */
    inline muGrid::RealField & get_native_stress() final;

    /**
     * returns a map on stored native stress field. Throws a runtime error if
     * native stress has not been stored
     */
    inline muGrid::MappedEigenField<Stress_t, Mapping::Mut, IterUnit::SubPt> &
    get_mapped_native_stress();

   protected:
    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <SplitCell is_cell_split, StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <SplitCell is_cell_split, StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P,
                                        muGrid::RealField & K)
        __attribute__((visibility("default")));

    muGrid::OptionalMappedField<
        muGrid::MappedEigenField<Stress_t, Mapping::Mut, IterUnit::SubPt>>
        native_stress;
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  MaterialMuSpectre<Material, DimM, Parent_>::MaterialMuSpectre(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent(name, spatial_dimension, DimM, nb_quad_pts,
               parent_field_collection),
        native_stress{*this->internal_fields,
                      this->get_prefix() + "native_stress", QuadPtTag} {
    static_assert(
        std::is_same<typename traits::StressMap_t::Scalar, Real>::value,
        "The stress map needs to be of type Real");
    static_assert(
        std::is_same<typename traits::StrainMap_t::Scalar, Real>::value,
        "The strain map needs to be of type Real");
    static_assert(
        std::is_same<typename traits::TangentMap_t::Scalar, Real>::value,
        "The tangent map needs to be of type Real");
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <class... ConstructorArgs>
  Material &
  MaterialMuSpectre<Material, DimM, Parent_>::make(std::shared_ptr<Cell> cell,
                                                   const std::string & name,
                                                   ConstructorArgs &&... args) {
    auto mat{std::make_unique<Material>(name, cell->get_spatial_dim(),
                                        cell->get_nb_quad_pts(), args...)};
    using traits = MaterialMuSpectre_traits<Material>;
    auto && form{cell->get_formulation()};
    constexpr StrainMeasure expected_strain_m{traits::strain_measure};
    if (form == Formulation::small_strain) {
      Parent::check_small_strain_capability(expected_strain_m);
    }

    auto & mat_ref{*mat};
    auto is_cell_split{cell->get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell->add_material(std::move(mat));
    return mat_ref;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <class... ConstructorArgs>
  Material & MaterialMuSpectre<Material, DimM, Parent_>::make(
      std::shared_ptr<CellData> cell_data, const std::string & name,
      ConstructorArgs &&... args) {
    if (not cell_data->has_nb_quad_pts()) {
      std::stringstream error_message{};
      error_message << "The number of quadrature points per pixel has not been "
        "set yet for this cell!";
      throw MaterialError{error_message.str()};
    }
    auto mat{std::make_unique<Material>(name, cell_data->get_spatial_dim(),
                                        cell_data->get_nb_quad_pts(), args...)};
    auto & mat_ref{*mat};
    cell_data->add_material(std::move(mat));
    return mat_ref;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <class... InternalArgs>
  void MaterialMuSpectre<Material, DimM, Parent_>::add_pixel_split(
      const size_t & pixel_id, Real ratio, InternalArgs... args) {
    auto & this_mat = static_cast<Material &>(*this);
    this_mat.add_pixel(pixel_id, args...);
    this->assigned_ratio->get_field().push_back(ratio);
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  void MaterialMuSpectre<Material, DimM, Parent_>::add_split_pixels_precipitate(
      const std::vector<size_t> & intersected_pixels,
      const std::vector<Real> & intersection_ratios) {
    // assign precipitate materials:
    for (auto && tup : akantu::zip(intersected_pixels, intersection_ratios)) {
      auto pix = std::get<0>(tup);
      auto ratio = std::get<1>(tup);
      this->add_pixel_split(pix, ratio);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <class... ConstructorArgs>
  std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>
  MaterialMuSpectre<Material, DimM, Parent_>::make_evaluator(
      ConstructorArgs &&... args) {
    constexpr Index_t SpatialDimension{DimM};
    constexpr Index_t NbQuadPts{1};
    auto mat = std::make_shared<Material>("name", SpatialDimension, NbQuadPts,
                                          args...);
    using Ret_t =
        std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>;
    return Ret_t(mat, MaterialEvaluator<DimM>{mat});
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  bool MaterialMuSpectre<Material, DimM, Parent_>::has_native_stress() const {
    return this->native_stress.has_value();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  muGrid::RealField &
  MaterialMuSpectre<Material, DimM, Parent_>::get_native_stress() {
    if (not this->native_stress.has_value()) {
      throw muGrid::RuntimeError("native stress has not been evaluated");
    }
    return this->native_stress.get().get_field();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  auto MaterialMuSpectre<Material, DimM, Parent_>::get_mapped_native_stress()
      -> muGrid::MappedEigenField<Stress_t, Mapping::Mut, IterUnit::SubPt> & {
    if (not this->native_stress.has_value()) {
      throw muGrid::RuntimeError("native stress has not been evaluated");
    }
    return this->native_stress.get();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  void MaterialMuSpectre<Material, DimM, Parent_>::compute_stresses(
      const muGrid::RealField & F, muGrid::RealField & P,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    switch (is_cell_split) {
    case (SplitCell::no):
      // fall-through;  laminate and whole pixels treated same at this point
    case (SplitCell::laminate): {
      switch (store_native_stress) {
      case StoreNativeStress::no: {
        this->compute_stresses_worker<SplitCell::no, StoreNativeStress::no>(F,
                                                                            P);
        break;
      }
      case StoreNativeStress::yes: {
        this->compute_stresses_worker<SplitCell::no, StoreNativeStress::yes>(F,
                                                                             P);
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown native stress treatment");
        break;
      }
      }
      break;
    }
    case (SplitCell::simple): {
      switch (store_native_stress) {
      case StoreNativeStress::no: {
        this->compute_stresses_worker<SplitCell::simple, StoreNativeStress::no>(
            F, P);
        break;
      }
      case StoreNativeStress::yes: {
        this->compute_stresses_worker<SplitCell::simple,
                                      StoreNativeStress::yes>(F, P);
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown native stress treatment");
        break;
      }
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown Splitness status");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  void MaterialMuSpectre<Material, DimM, Parent_>::compute_stresses_tangent(
      const muGrid::RealField & F, muGrid::RealField & P, muGrid::RealField & K,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    switch (is_cell_split) {
    case (SplitCell::no):
      // fall-through;  laminate and whole pixels treated same at this point
    case (SplitCell::laminate): {
      switch (store_native_stress) {
      case StoreNativeStress::no: {
        this->compute_stresses_worker<SplitCell::no, StoreNativeStress::no>(
            F, P, K);
        break;
      }
      case StoreNativeStress::yes: {
        this->compute_stresses_worker<SplitCell::no, StoreNativeStress::yes>(
            F, P, K);
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown native stress treatment");
        break;
      }
      }
      break;
    }
    case (SplitCell::simple): {
      switch (store_native_stress) {
      case StoreNativeStress::no: {
        this->compute_stresses_worker<SplitCell::simple, StoreNativeStress::no>(
            F, P, K);
        break;
      }
      case StoreNativeStress::yes: {
        this->compute_stresses_worker<SplitCell::simple,
                                      StoreNativeStress::yes>(F, P, K);
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown native stress treatment");
        break;
      }
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown Splitness status");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <SplitCell IsCellSplit, StoreNativeStress DoStoreNative>
  void MaterialMuSpectre<Material, DimM, Parent_>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P,
      muGrid::RealField & K) {
    auto & this_mat{static_cast<Material &>(*this)};

    using iterable_proxy_t = iterable_proxy<
        std::tuple<typename traits::StrainMap_t>,
        std::tuple<typename traits::StressMap_t, typename traits::TangentMap_t>,
        IsCellSplit>;

    iterable_proxy_t fields(*this, F, P, K);

    auto * native_stress_map{(DoStoreNative == StoreNativeStress::yes)
                                 ? &this->native_stress.get().get_map()
                                 : nullptr};
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
      auto && strain{std::get<0>(std::get<0>(arglist))};
      auto && stress_stiffness{std::get<1>(arglist)};
      auto && stress{std::get<0>(stress_stiffness)};
      auto && stiffness{std::get<1>(stress_stiffness)};
      auto && quad_pt_id{std::get<2>(arglist)};

      auto && stress_stiffness_result{
          this_mat.evaluate_stress_tangent(strain, quad_pt_id)};
      auto && stress_result{std::get<0>(stress_stiffness_result)};
      auto && stiffness_result{std::get<1>(stress_stiffness_result)};

      if (DoStoreNative == StoreNativeStress::yes) {
        auto && quad_pt_native_stress{(*native_stress_map)[quad_pt_id]};
        quad_pt_native_stress = stress_result;
      }
      if (IsCellSplit == SplitCell::simple) {
        auto && ratio{std::get<3>(arglist)};
        stress += ratio * stress_result;
        stiffness += ratio * stiffness_result;
      } else {
        stress = stress_result;
        stiffness = stiffness_result;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  template <SplitCell IsCellSplit, StoreNativeStress DoStoreNative>
  void MaterialMuSpectre<Material, DimM, Parent_>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P) {
    auto & this_mat = static_cast<Material &>(*this);

    using iterable_proxy_t =
        iterable_proxy<std::tuple<typename traits::StrainMap_t>,
                       std::tuple<typename traits::StressMap_t>, IsCellSplit>;

    iterable_proxy_t fields(*this, F, P);

    auto * native_stress_map{(DoStoreNative == StoreNativeStress::yes)
                                 ? &this->native_stress.get().get_map()
                                 : nullptr};
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

      auto && strain{std::get<0>(std::get<0>(arglist))};
      auto && stress{std::get<0>(std::get<1>(arglist))};
      auto && quad_pt_id{std::get<2>(arglist)};

      auto && stress_result{this_mat.evaluate_stress(strain, quad_pt_id)};

      if (DoStoreNative == StoreNativeStress::yes) {
        auto && quad_pt_native_stress{(*native_stress_map)[quad_pt_id]};
        quad_pt_native_stress = stress_result;
      }

      if (IsCellSplit == SplitCell::simple) {
        auto && ratio{std::get<3>(arglist)};
        stress += ratio * stress_result;
      } else {
        stress = stress_result;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM, class Parent_>
  auto MaterialMuSpectre<Material, DimM, Parent_>::constitutive_law_dynamic(
      const Eigen::Ref<const DynMatrix_t> & strain,
      const size_t & quad_pt_index) -> std::tuple<DynMatrix_t, DynMatrix_t> {
    auto & this_mat = static_cast<Material &>(*this);

    if ((strain.rows() != Strain_t::RowsAtCompileTime) or
        (strain.cols() != Strain_t::ColsAtCompileTime)) {
      std::stringstream err_msg{};
      err_msg << "Shape mismatch: expected an input strain of shape ("
              << Strain_t::RowsAtCompileTime << ", "
              << Strain_t::ColsAtCompileTime << "), but got (" << strain.rows()
              << ", " << strain.cols() << ").";
      throw MaterialError{err_msg.str()};
    }

    Eigen::Map<const Strain_t> F{strain.data()};

    if (strain.cols() != DimM or strain.rows() != DimM) {
      std::stringstream error{};
      error << "incompatible strain shape, expected " << DimM << " × " << DimM
            << ", but received " << strain.rows() << " × " << strain.cols()
            << "." << std::endl;
      throw MaterialError(error.str());
    }

    std::tuple<Stress_t, Tangent_t> PK{
        this_mat.evaluate_stress_tangent(F, quad_pt_index)};
    return PK;
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_MUSPECTRE_HH_
