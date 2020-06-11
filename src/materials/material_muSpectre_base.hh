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
#include "materials/iterable_proxy.hh"

#include "cell/cell.hh"

#include "libmugrid/field_map_static.hh"

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

  template <class Material, Index_t DimM>
  class MaterialMuSpectre;

  /**
   * Base class for most convenient implementation of materials
   */
  template <class Material, Index_t DimM>
  class MaterialMuSpectre : public MaterialBase {
   public:
    /**
     * type used to determine whether the
     * `muSpectre::MaterialBase::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = MatTB::NeedTangent;
    using Parent = MaterialBase;  //!< base class

    //! traits for the CRTP subclass
    using traits = MaterialMuSpectre_traits<Material>;
    using DynMatrix_t = Parent::DynMatrix_t;

    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stress_t = Strain_t;
    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

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
    inline static Material & make(Cell & cell, const std::string & name,
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
                     const Formulation & form,
                     const SplitCell & is_cell_split = SplitCell::no,
                     const StoreNativeStress & store_native_stress =
                         StoreNativeStress::no) final;

    //! computes stress and tangent modulus
    inline void
    compute_stresses_tangent(const muGrid::RealField & F, muGrid::RealField & P,
                             muGrid::RealField & K, const Formulation & form,
                             const SplitCell & is_cell_split = SplitCell::no,
                             const StoreNativeStress & store_native_stress =
                                 StoreNativeStress::no) final;

    //! return the material dimension at compile time
    constexpr static Index_t MaterialDimension() { return DimM; }

    inline std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & pixel_index,
                             const Formulation & form) final;
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
    inline muGrid::MappedT2Field<Real, Mapping::Mut, DimM,
                                 IterUnit::SubPt> &
    get_mapped_native_stress();

   protected:
    //! dispatches the correct compute_stresses worker
    template <Formulation Form, SplitCell IsSplit, class... Args>
    inline void
    compute_stresses_dispatch1(const StoreNativeStress store_native_stress,
                               Args &&... args);

    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split,
              StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split,
              StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P,
                                        muGrid::RealField & K)
        __attribute__((visibility("default")));

    muGrid::OptionalMappedField<
        muGrid::MappedT2Field<Real, Mapping::Mut, DimM, IterUnit::SubPt>>
        native_stress;
  };

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  MaterialMuSpectre<Material, DimM>::MaterialMuSpectre(
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
  template <class Material, Index_t DimM>
  template <class... ConstructorArgs>
  Material &
  MaterialMuSpectre<Material, DimM>::make(Cell & cell, const std::string & name,
                                          ConstructorArgs &&... args) {
    auto mat = std::make_unique<Material>(name, cell.get_spatial_dim(),
                                          cell.get_nb_quad_pts(), args...);
    using traits = MaterialMuSpectre_traits<Material>;
    auto && form = cell.get_formulation();
    constexpr StrainMeasure expected_strain_m{traits::strain_measure};
    if (form == Formulation::small_strain) {
      check_small_strain_capability(expected_strain_m);
    }

    auto & mat_ref = *mat;
    auto is_cell_split{cell.get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell.add_material(std::move(mat));
    return mat_ref;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <class... InternalArgs>
  void MaterialMuSpectre<Material, DimM>::add_pixel_split(
      const size_t & pixel_id, Real ratio, InternalArgs... args) {
    auto & this_mat = static_cast<Material &>(*this);
    this_mat.add_pixel(pixel_id, args...);
    this->assigned_ratio->get_field().push_back(ratio);
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  void MaterialMuSpectre<Material, DimM>::add_split_pixels_precipitate(
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
  template <class Material, Index_t DimM>
  template <class... ConstructorArgs>
  std::tuple<std::shared_ptr<Material>, MaterialEvaluator<DimM>>
  MaterialMuSpectre<Material, DimM>::make_evaluator(
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
  template <class Material, Index_t DimM>
  bool MaterialMuSpectre<Material, DimM>::has_native_stress() const {
    return this->native_stress.has_value();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  muGrid::RealField & MaterialMuSpectre<Material, DimM>::get_native_stress() {
    if (not this->native_stress.has_value()) {
      throw muGrid::RuntimeError("native stress has not been evaluated");
    }
    return this->native_stress.get().get_field();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  auto MaterialMuSpectre<Material, DimM>::get_mapped_native_stress()
      -> muGrid::MappedT2Field<Real, Mapping::Mut, DimM,
                               IterUnit::SubPt> & {
    if (not this->native_stress.has_value()) {
      throw muGrid::RuntimeError("native stress has not been evaluated");
    }
    return this->native_stress.get();
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <Formulation Form, SplitCell IsSplit, class... Args>
  void MaterialMuSpectre<Material, DimM>::compute_stresses_dispatch1(
      const StoreNativeStress store_native_stress, Args &&... args) {
    switch (store_native_stress) {
    case StoreNativeStress::yes: {
      this->template compute_stresses_worker<Form, IsSplit,
                                             StoreNativeStress::yes>(
          std::forward<Args>(args)...);
      break;
    }
    case StoreNativeStress::no: {
      this->template compute_stresses_worker<Form, IsSplit,
                                             StoreNativeStress::no>(
          std::forward<Args>(args)...);
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown value for store native stress");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  void MaterialMuSpectre<Material, DimM>::compute_stresses(
      const muGrid::RealField & F, muGrid::RealField & P,
      const Formulation & form, const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    using traits = MaterialMuSpectre_traits<Material>;
    constexpr StrainMeasure expected_strain_m{traits::strain_measure};

    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
        // fall-through;  laminate and whole pixels treated same at this point
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::finite_strain,
                                         SplitCell::no>(store_native_stress, F,
                                                        P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::finite_strain,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      check_small_strain_capability(expected_strain_m);
      switch (is_cell_split) {
      case (SplitCell::no):
        // fall-through;  laminate and whole pixels treated same at this point
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::small_strain,
                                         SplitCell::no>(store_native_stress, F,
                                                        P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::small_strain,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::native: {
      switch (is_cell_split) {
      case (SplitCell::no):
        // fall-through; laminate and whole pixels treated same at this point
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::native, SplitCell::no>(
            store_native_stress, F, P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::native,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  void MaterialMuSpectre<Material, DimM>::compute_stresses_tangent(
      const muGrid::RealField & F, muGrid::RealField & P, muGrid::RealField & K,
      const Formulation & form, const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    using traits = MaterialMuSpectre_traits<Material>;
    constexpr StrainMeasure expected_strain_m{traits::strain_measure};

    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
        // fall-through;  laminate and whole pixels treated same at this point
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::finite_strain,
                                         SplitCell::no>(store_native_stress, F,
                                                        P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::finite_strain,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P, K);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      check_small_strain_capability(expected_strain_m);
      switch (is_cell_split) {
      case (SplitCell::no):
        // fall-through;  laminate and whole pixels treated same at this point
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::small_strain,
                                         SplitCell::no>(store_native_stress, F,
                                                        P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::small_strain,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P, K);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::native: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_dispatch1<Formulation::native, SplitCell::no>(
            store_native_stress, F, P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_dispatch1<Formulation::native,
                                         SplitCell::simple>(store_native_stress,
                                                            F, P, K);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <Formulation Form, SplitCell IsCellSplit,
            StoreNativeStress DoStoreNative>
  void MaterialMuSpectre<Material, DimM>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P,
      muGrid::RealField & K) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    auto & this_mat{static_cast<Material &>(*this)};
    using traits = MaterialMuSpectre_traits<Material>;
    constexpr StrainMeasure expected_strain_m{traits::strain_measure};
    if (Form == Formulation::small_strain) {
      check_small_strain_capability(expected_strain_m);
    }

    using iterable_proxy_t = iterable_proxy<
        std::tuple<typename traits::StrainMap_t>,
        std::tuple<typename traits::StressMap_t, typename traits::TangentMap_t>,
        IsCellSplit>;

    iterable_proxy_t fields(*this, F, P, K);

    switch (DoStoreNative) {
    case StoreNativeStress::no: {
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
        auto && quad_pt_id{std::get<2>(arglist)};

        if (IsCellSplit == SplitCell::simple) {
          auto && ratio{std::get<3>(arglist)};
          MatTB::constitutive_law_tangent<Form>(
              this_mat, strain, stress_stiffness, quad_pt_id, ratio);
        } else {
          MatTB::constitutive_law_tangent<Form>(this_mat, strain,
                                                stress_stiffness, quad_pt_id);
        }
      }
      break;
    }
    case StoreNativeStress::yes: {
      auto & native_stress_map{this->native_stress.get().get_map()};
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
        auto && quad_pt_id{std::get<2>(arglist)};
        auto && quad_pt_native_stress{native_stress_map[quad_pt_id]};

        if (IsCellSplit == SplitCell::simple) {
          auto && ratio{std::get<3>(arglist)};
          MatTB::constitutive_law_tangent<Form>(this_mat, strain,
                                                stress_stiffness, quad_pt_id,
                                                ratio, quad_pt_native_stress);
        } else {
          MatTB::constitutive_law_tangent<Form>(this_mat, strain,
                                                stress_stiffness, quad_pt_id,
                                                quad_pt_native_stress);
        }
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown StoreNativeStress parameter");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <Formulation Form, SplitCell IsCellSplit,
            StoreNativeStress DoStoreNative>
  void MaterialMuSpectre<Material, DimM>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems
    */
    auto & this_mat = static_cast<Material &>(*this);

    using iterable_proxy_t =
        iterable_proxy<std::tuple<typename traits::StrainMap_t>,
                       std::tuple<typename traits::StressMap_t>, IsCellSplit>;

    iterable_proxy_t fields(*this, F, P);

    switch (DoStoreNative) {
    case StoreNativeStress::no: {
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
        auto && quad_pt_id{std::get<2>(arglist)};

        if (IsCellSplit == SplitCell::simple) {
          auto && ratio{std::get<3>(arglist)};
          MatTB::constitutive_law<Form>(this_mat, strain, stress, quad_pt_id,
                                        ratio);
        } else {
          MatTB::constitutive_law<Form>(this_mat, strain, stress, quad_pt_id);
        }
      }
      break;
    }
    case StoreNativeStress::yes: {
      auto & native_stress_map{this->native_stress.get().get_map()};

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
        auto && quad_pt_id{std::get<2>(arglist)};
        auto && quad_pt_native_stress{native_stress_map[quad_pt_id]};

        if (IsCellSplit == SplitCell::simple) {
          auto && ratio{std::get<3>(arglist)};
          MatTB::constitutive_law<Form>(this_mat, strain, stress, quad_pt_id,
                                        ratio, quad_pt_native_stress);
        } else {
          MatTB::constitutive_law<Form>(this_mat, strain, stress, quad_pt_id,
                                        quad_pt_native_stress);
        }
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown StoreNativeStress parameter");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  auto MaterialMuSpectre<Material, DimM>::constitutive_law_dynamic(
      const Eigen::Ref<const DynMatrix_t> & strain,
      const size_t & quad_pt_index, const Formulation & form)
      -> std::tuple<DynMatrix_t, DynMatrix_t> {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());
    std::tuple<Stress_t, Stiffness_t> PK{};

    if (strain.cols() != DimM or strain.rows() != DimM) {
      std::stringstream error{};
      error << "incompatible strain shape, expected " << DimM << " × " << DimM
            << ", but received " << strain.rows() << " × " << strain.cols()
            << "." << std::endl;
      throw MaterialError(error.str());
    }

    switch (form) {
    case Formulation::finite_strain: {
      MatTB::constitutive_law_tangent<Formulation::finite_strain>(
          this_mat, std::make_tuple(F), PK, quad_pt_index);
      break;
    }
    case Formulation::small_strain: {
      MatTB::constitutive_law_tangent<Formulation::small_strain>(
          this_mat, std::make_tuple(F), PK, quad_pt_index);
      break;
    }
    default:
      throw MaterialError("Unknown formulation");
      break;
    }
    return PK;
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_MUSPECTRE_BASE_HH_
