/**
 * @file   material_laminate.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date    04 Jun 2018
 *
 * @brief material that uses laminae homogenisation
 *
 * Copyright © 2018 Ali Falsafi
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
#include "materials/materials_toolbox.hh"
#include "materials/material_evaluator.hh"
#include "materials/laminate_homogenisation.hh"
#include "common/intersection_octree.hh"

#include "cell/cell.hh"

#include "libmugrid/T4_map_proxy.hh"

#include <vector>
namespace muSpectre {
  template <Index_t DimM>
  class MaterialLaminate;

  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLaminate<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<Real, Mapping::Const, DimM, PixelSubDiv::QuadPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, PixelSubDiv::QuadPt>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimM, PixelSubDiv::QuadPt>;

    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK1};
  };

  template <Index_t DimM>
  class MaterialLaminate : public MaterialBase {
   public:
    //! base class
    using Parent = MaterialBase;
    using RealField = muGrid::RealField;
    using DynMatrix_t = Parent::DynMatrix_t;
    //
    using MatBase_t = MaterialBase;
    using MatPtr_t = std::shared_ptr<MatBase_t>;

    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    using VectorField_t = muGrid::RealField;
    using MappedVectorField_t =
        muGrid::MappedT1Field<Real, Mapping::Mut, DimM, PixelSubDiv::QuadPt>;
    using VectorFieldMap_t =
        muGrid::T1FieldMap<Real, Mapping::Mut, DimM, PixelSubDiv::QuadPt>;

    using ScalarField_t = muGrid::RealField;
    using MappedScalarField_t =
        muGrid::MappedScalarField<Real, Mapping::Mut, PixelSubDiv::QuadPt>;
    using ScalarFieldMap_t =
        muGrid::ScalarFieldMap<Real, Mapping::Mut, PixelSubDiv::QuadPt>;

    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stress_t = Strain_t;
    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = MatTB::NeedTangent;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLaminate>;

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

    //! Factory
    static MaterialLaminate<DimM> & make(Cell & cell, const std::string & name);

    template <class... ConstructorArgs>
    static std::tuple<std::shared_ptr<MaterialLaminate<DimM>>,
                      MaterialEvaluator<DimM>>
    make_evaluator(ConstructorArgs &&... args);

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */

    template <typename Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & pixel_index,
                                          const Formulation & form);

    /**
     * evaluates second Piola-Kirchhoff stress and its corresponding tangent
     * given the Green-Lagrange strain (or Cauchy stress and its corresponding
     * tangetn if called with a small strain tensor)
     */
    template <typename Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & pixel_index,
                            const Formulation & form);

    template <Formulation Form, class Strains, class Stresses>
    void constitutive_law(const Strains & strains, Stresses & stress,
                          const size_t & quad_pt_id);

    template <Formulation Form, class Strains, class Stresses>
    void constitutive_law(const Strains & strains, Stresses & stress,
                          const size_t & quad_pt_id, const Real & ratio);

    template <Formulation Form, class Strains, class Stresses>
    void constitutive_law_tangent(const Strains & strains, Stresses & stresses,
                                  const size_t & quad_pt_id);

    template <Formulation Form, class Strains, class Stresses>
    void constitutive_law_tangent(const Strains & strains, Stresses & stresses,
                                  const size_t & quad_pt_id,
                                  const Real & ratio);

    template <Formulation Form, class Strains_t>
    decltype(auto) constitutive_law(const Strains_t & Strains,
                                    const size_t & quad_pt_id);

    template <Formulation Form, class Strains_t>
    decltype(auto) constitutive_law_tangent(const Strains_t & Strains,
                                            const size_t & quad_pt_id);

    //! computes stress
    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;
    void compute_stresses(const RealField & F, RealField & P,
                          const Formulation & form,
                          const SplitCell & is_cell_split,
                          const StoreNativeStress & store_native_stress) final;
    //!  stress and tangent modulus
    void compute_stresses_tangent(
        const RealField & F, RealField & P, RealField & K,
        const Formulation & form, const SplitCell & is_cell_split,
        const StoreNativeStress & store_native_stress) final;
    /**
     * overload add_pixel to write into volume ratio and normal vectors and ...
     */
    void add_pixel(const size_t & pixel_id) final;

    /**
     * overload add_pixel to add underlying materials and their ratio and
     * interface direction to the material lamiante
     */
    void
    add_pixel(const size_t & pixel_id, MatPtr_t mat1, MatPtr_t mat2,
              const Real & ratio,
              const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
                  normal_Vector);

    /**
     * This function adds pixels according to the precipitate intersected pixels
     * and the materials incolved
     */
    void add_pixels_precipitate(
        const std::vector<Ccoord_t<DimM>> & intersected_pixels,
        const std::vector<Index_t> & intersected_pixels_id,
        const std::vector<Real> & intersection_ratios,
        const std::vector<Eigen::Matrix<Real, DimM, 1>> & intersection_normals,
        MatPtr_t mat1, MatPtr_t mat2);

    std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & pixel_index,
                             const Formulation & form) final;

    /**
     * @brief      wrapper around the evaluate_tress function to handle the
     * operation on the the evaluated stress and tangent by the material(s)
     */
    template <Formulation Form, class Strain, class Stress, class Op,
              class NativeTreat>
    void evaluate_material_stress(Strain && strain, Stress & stress,
                                  const size_t & quad_pt_id,
                                  const Op & operation,
                                  NativeTreat & native_stress_treatment);
    /**
     * @brief      wrapper around the evaluate_tress function to handle the
     * operation on the the evaluated stress and tangent by the material(s)
     */

    template <Formulation Form, class Strain, class Stress, class Stiffness,
              class Op, class NativeTreat>
    void evaluate_material_stress_tangent(
        Strain && strain, std::tuple<Stress, Stiffness> & stress_stiffness,
        const size_t & quad_pt_id, const Op & operation,
        NativeTreat & native_stress_treatment);

   protected:
    MappedVectorField_t
        normal_vector_field;  //!< field holding the normal vector
                              //!< of the interface of the layers

    MappedScalarField_t
        volume_ratio_field;  //!< field holding the normal vector

    std::vector<MatPtr_t> material_left_vector{};
    std::vector<MatPtr_t> material_right_vector{};

    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell IsCellSplit>
    inline void compute_stresses_worker(const RealField & F, RealField & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell IsCellSplit>
    inline void compute_stresses_worker(const RealField & F, RealField & P,
                                        RealField & K)
        __attribute__((visibility("default")));
  };

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <class Strain>
  decltype(auto)
  MaterialLaminate<DimM>::evaluate_stress(const Eigen::MatrixBase<Strain> & E,
                                          const size_t & pixel_index,
                                          const Formulation & form) {
    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;
    auto && mat_l{material_left_vector[pixel_index]};
    auto && mat_r{material_right_vector[pixel_index]};

    Strain_t E_eval(E);

    const Function_t mat_l_evaluate_stress_tangent_func{
        [&mat_l, &pixel_index, &form](const Eigen::Ref<const Strain_t> & E) {
          return mat_l->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 form);
        }};

    const Function_t mat_r_evaluate_stress_tangent_func{
        [&mat_r, &pixel_index, &form](const Eigen::Ref<const Strain_t> & E) {
          return mat_r->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 form);
        }};

    auto && ratio{this->volume_ratio_field[pixel_index]};
    auto && normal_vec{this->normal_vector_field[pixel_index]};
    switch (form) {
    case Formulation::finite_strain: {
      return LamHomogen<DimM, Formulation::finite_strain>::evaluate_stress(
          E_eval, mat_l_evaluate_stress_tangent_func,
          mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    case Formulation::small_strain: {
      return LamHomogen<DimM, Formulation::small_strain>::evaluate_stress(
          E_eval, mat_l_evaluate_stress_tangent_func,
          mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    case Formulation::native: {
      return LamHomogen<DimM, Formulation::native>::evaluate_stress(
          E_eval, mat_l_evaluate_stress_tangent_func,
          mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    default: {
      throw muGrid::RuntimeError("Unknown formualtion");
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Strain>
  decltype(auto) MaterialLaminate<DimM>::evaluate_stress_tangent(
      const Eigen ::MatrixBase<Strain> & E, const size_t & pixel_index,
      const Formulation & form) {
    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;
    auto && mat_l{material_left_vector[pixel_index]};
    auto && mat_r{material_right_vector[pixel_index]};
    Strain_t E_eval(E);

    Function_t mat_l_evaluate_stress_tangent_func{
        [&mat_l, &pixel_index, &form](const Eigen::Ref<const Strain_t> & E) {
          return mat_l->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 form);
        }};

    Function_t mat_r_evaluate_stress_tangent_func{
        [&mat_r, &pixel_index, &form](const Eigen::Ref<const Strain_t> & E) {
          return mat_r->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 form);
        }};

    std::tuple<Stress_t, Stiffness_t> ret_stress_stiffness{};
    auto && ratio{this->volume_ratio_field[pixel_index]};
    auto && normal_vec{this->normal_vector_field[pixel_index]};
    switch (form) {
    case Formulation::finite_strain: {
      return LamHomogen<DimM, Formulation::finite_strain>::
          evaluate_stress_tangent(E_eval, mat_l_evaluate_stress_tangent_func,
                                  mat_r_evaluate_stress_tangent_func, ratio,
                                  normal_vec);
      break;
    }
    case Formulation::small_strain: {
      return LamHomogen<DimM, Formulation::small_strain>::
          evaluate_stress_tangent(E_eval, mat_l_evaluate_stress_tangent_func,
                                  mat_r_evaluate_stress_tangent_func, ratio,
                                  normal_vec);
      break;
    }
    case Formulation::native: {
      return LamHomogen<DimM, Formulation::native>::evaluate_stress_tangent(
          E_eval, mat_l_evaluate_stress_tangent_func,
          mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    default: {
      throw muGrid::RuntimeError("Unknown formualtion");
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <Formulation Form, SplitCell IsCellSplit>
  void MaterialLaminate<DimM>::compute_stresses_worker(const RealField & F,
                                                       RealField & P,
                                                       RealField & K) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */

    using traits = MaterialMuSpectre_traits<MaterialLaminate<DimM>>;

    using iterable_proxy_t = iterable_proxy<
        std::tuple<typename traits::StrainMap_t>,
        std::tuple<typename traits::StressMap_t, typename traits::TangentMap_t>,
        IsCellSplit>;

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
      auto && index{std::get<2>(arglist)};
      if (IsCellSplit == SplitCell::simple) {
        auto && ratio{std::get<3>(arglist)};
        this->constitutive_law_tangent<Form>(strain, stress_stiffness, index,
                                             ratio);
      } else {
        this->constitutive_law_tangent<Form>(strain, stress_stiffness, index);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <Formulation Form, SplitCell IsCellSplit>
  void MaterialLaminate<DimM>::compute_stresses_worker(const RealField & F,
                                                       RealField & P) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems
    */
    using traits = MaterialMuSpectre_traits<MaterialLaminate<DimM>>;

    using iterable_proxy_t =
        iterable_proxy<std::tuple<typename traits::StrainMap_t>,
                       std::tuple<typename traits::StressMap_t>, IsCellSplit>;

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
      auto && quad_pt_id{std::get<2>(arglist)};
      if (IsCellSplit == SplitCell::simple) {
        auto && ratio{std::get<3>(arglist)};
        this->constitutive_law<Form>(strain, stress, quad_pt_id, ratio);
      } else {
        this->constitutive_law<Form>(strain, stress, quad_pt_id);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class... ConstructorArgs>
  std::tuple<std::shared_ptr<MaterialLaminate<DimM>>, MaterialEvaluator<DimM>>
  MaterialLaminate<DimM>::make_evaluator(ConstructorArgs &&... args) {
    constexpr Index_t SpatialDimension{DimM};
    constexpr Index_t NbQuadPts{1};
    auto mat{std::make_shared<MaterialLaminate<DimM>>("name", SpatialDimension,
                                                      NbQuadPts, args...)};
    using Ret_t = std::tuple<std::shared_ptr<MaterialLaminate<DimM>>,
                             MaterialEvaluator<DimM>>;
    return Ret_t(mat, MaterialEvaluator<DimM>{mat});
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialLaminate<DimM>::constitutive_law_dynamic(
      const Eigen::Ref<const DynMatrix_t> & strain,
      const size_t & quad_pt_index, const Formulation & form)
      -> std::tuple<DynMatrix_t, DynMatrix_t> {
    Eigen::Map<const Strain_t> F(strain.data());

    Stress_t P{};
    Stiffness_t K{};
    std::tuple<Stress_t, Stiffness_t> PK{std::make_tuple(P, K)};

    if (strain.cols() != DimM or strain.rows() != DimM) {
      std::stringstream error{};
      error << "incompatible strain shape, expected " << DimM << " × " << DimM
            << ", but received " << strain.rows() << " × " << strain.cols()
            << "." << std::endl;
      throw MaterialError(error.str());
    }

    switch (form) {
    case Formulation::finite_strain: {
      this->constitutive_law_tangent<Formulation::finite_strain>(
          std::make_tuple(F), PK, quad_pt_index);
      return PK;
      break;
    }
    case Formulation::small_strain: {
      this->constitutive_law_tangent<Formulation::small_strain>(
          std::make_tuple(F), PK, quad_pt_index);
      return PK;
      break;
    }
    default:
      throw MaterialError("Unknown formulation");
      break;
    }
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <Formulation Form, class Strain, class Stress, class Op,
            class NativeTreat>
  void MaterialLaminate<DimM>::evaluate_material_stress(
      Strain && strain, Stress & stress, const size_t & quad_pt_id,
      const Op & operation, NativeTreat & native_stress_treatment) {
    using traits = MaterialMuSpectre_traits<MaterialLaminate<DimM>>;

    constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
    constexpr StrainMeasure expected_strain_m{
        get_formulation_strain_type(Form, traits::strain_measure)};

    switch (Form) {
    case Formulation::small_strain: {
      auto && eps{MatTB::convert_strain<stored_strain_m, expected_strain_m>(
          std::get<0>(strain))};

      auto && stress_result{this->evaluate_stress(eps, quad_pt_id, Form)};
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
      auto && stress_result{
          this->evaluate_stress(std::move(E), quad_pt_id, Form)};

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

      operation(
          this->evaluate_stress(std::move(strain_converted), quad_pt_id, Form),
          stress);
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown formulation");
      break;
    }
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <Formulation Form, class Strains, class Stresses>
  void MaterialLaminate<DimM>::constitutive_law(const Strains & strains,
                                                Stresses & stresses,
                                                const size_t & quad_pt_id,
                                                const Real & ratio) {
    MatTB::OperationAddition operation_addition(ratio);
    MatTB::NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
    this->evaluate_material_stress<Form>(strains, stresses, quad_pt_id,
                                         operation_addition, stress_treatment);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <Formulation Form, class Strains, class Stresses>
  void MaterialLaminate<DimM>::constitutive_law(const Strains & strains,
                                                Stresses & stresses,
                                                const size_t & quad_pt_id) {
    MatTB::OperationAssignment operation_assignment;
    MatTB::NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
    this->evaluate_material_stress<Form>(
        strains, stresses, quad_pt_id, operation_assignment, stress_treatment);
  }

  /* ----------------------------------------------------------------------*/

  template <Index_t DimM>
  template <Formulation Form, class Strain, class Stress, class Stiffness,
            class Op, class NativeTreat>
  void MaterialLaminate<DimM>::evaluate_material_stress_tangent(
      Strain && strain, std::tuple<Stress, Stiffness> & stress_stiffness,
      const size_t & quad_pt_id, const Op & operation,
      NativeTreat & native_stress_treatment) {
    using traits = MaterialMuSpectre_traits<MaterialLaminate<DimM>>;

    constexpr StrainMeasure stored_strain_m{get_stored_strain_type(Form)};
    constexpr StrainMeasure expected_strain_m{
        get_formulation_strain_type(Form, traits::strain_measure)};
    switch (Form) {
    case Formulation::small_strain: {
      auto && eps{MatTB::convert_strain<stored_strain_m, expected_strain_m>(
          std::get<0>(strain))};

      auto && stress_stiffness_mat{
          this->evaluate_stress_tangent(std::move(eps), quad_pt_id, Form)};
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
          this->evaluate_stress_tangent(std::move(E), quad_pt_id, Form)};

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
      auto && stress_stiffness_mat{this->evaluate_stress_tangent(
          std::move(strain_converted), quad_pt_id, Form)};
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

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <Formulation Form, class Strains, class Stresses>
  void MaterialLaminate<DimM>::constitutive_law_tangent(
      const Strains & strains, Stresses & stresses, const size_t & quad_pt_id,
      const Real & ratio) {
    MatTB::OperationAddition operation_addition(ratio);
    MatTB::NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
    this->evaluate_material_stress_tangent<Form>(
        strains, stresses, quad_pt_id, operation_addition, stress_treatment);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <Formulation Form, class Strains, class Stresses>
  void MaterialLaminate<DimM>::constitutive_law_tangent(
      const Strains & strains, Stresses & stresses, const size_t & quad_pt_id) {
    MatTB::OperationAssignment operation_assignment;
    MatTB::NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
    this->evaluate_material_stress_tangent<Form>(
        strains, stresses, quad_pt_id, operation_assignment, stress_treatment);
  }
  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre
#endif  // SRC_MATERIALS_MATERIAL_LAMINATE_HH_
