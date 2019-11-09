/**
 * @file   material_laminate.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   19 Oct 2018
 *
 * @brief  Defenition of MaterialLamiante class which is a lamiante
 * approximation constitutive law for two underlying materials with arbitrary
 * constutive law
 *
 * Copyright © 2019 Till Junge, Ali Falsafi
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

#include "libmugrid/T4_map_proxy.hh"
#include "common/muSpectre_common.hh"
#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "materials/material_evaluator.hh"
#include "materials/laminate_homogenisation.hh"
#include "common/intersection_octree.hh"
#include "cell/cell_base.hh"

#include <vector>
namespace muSpectre {
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLaminate;

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLaminate<DimS, DimM>> {
    using Parent = MaterialMuSpectre_traits<void>;  //!< base for elasticity

    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;
    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK1};
  };

  template <Dim_t DimS, Dim_t DimM>
  class MaterialLaminate : public MaterialBase<DimS, DimM> {
   public:
    //! base class
    using Parent = MaterialBase<DimS, DimM>;
    //! expected type for stress fields
    using StressField_t = typename Parent::StressField_t;
    //! expected type for strain fields
    using StrainField_t = typename Parent::StrainField_t;
    //! expected type for tangent stiffness fields
    using TangentField_t = typename Parent::TangentField_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    //
    using MatBase_t = MaterialBase<DimS, DimM>;
    using MatPtr_t = std::shared_ptr<MatBase_t>;

    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! local field collection used for internals
    using LFieldColl_t = muGrid::LocalFieldCollection<DimS>;

    using VectorField_t =
        muGrid::TensorField<LFieldColl_t, Real, firstOrder, DimM>;
    using VectorFieldMap_t =
        muGrid::MatrixFieldMap<LFieldColl_t, Real, DimM, 1>;

    using ScalarField_t = muGrid::ScalarField<LFieldColl_t, Real>;
    using ScalarFieldMap_t = muGrid::ScalarFieldMap<LFieldColl_t, Real>;

    using ScalarFieldDim_t = muGrid::ScalarField<LFieldColl_t, Dim_t>;
    using ScalarFieldMapDim_t = muGrid::ScalarFieldMap<LFieldColl_t, Dim_t>;

    using FieldMap_t = muGrid::MatrixFieldMap<LFieldColl_t, Real, DimM, DimM>;

    using DynMatrix_t = Parent::DynMatrix_t;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = MatTB::NeedTangent;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLaminate>;

    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for tangent stiffness fields
    using Tangent_t = muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! Default constructor
    MaterialLaminate() = delete;

    //! Constructor with name and material properties
    explicit MaterialLaminate(std::string name);

    //! Copy constructor
    MaterialLaminate(const MaterialLaminate & other) = delete;

    //! Move constructor
    MaterialLaminate(MaterialLaminate && other) = delete;

    //! Destructor
    virtual ~MaterialLaminate() = default;

    //! Factory
    static MaterialLaminate<DimS, DimM> & make(CellBase<DimS, DimM> & cell,
                                               std::string name);

    template <class... ConstructorArgs>
    static std::tuple<std::shared_ptr<MaterialLaminate<DimS, DimM>>,
                      MaterialEvaluator<DimM>>
    make_evaluator(ConstructorArgs &&... args);
    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */

    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E, const size_t & pixel_index,
                                          Formulation form);

    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t && E,
                                                  const size_t & pixel_index,
                                                  Formulation form);

    std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & pixel_index,
                             const Formulation & form) final;

    //! computes stress
    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;
    void compute_stresses(const StrainField_t & F, StressField_t & P,
                          Formulation form, SplitCell is_cell_split) final;
    //! computes stress and tangent modulus
    void compute_stresses_tangent(const StrainField_t & F, StressField_t & P,
                                  TangentField_t & K, Formulation form,
                                  SplitCell is_cell_split) final;
    /**
     * overload add_pixel to write into volume ratio and normal vectors and ...
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(
        const Ccoord_t<DimS> & pixel, MatPtr_t mat1, MatPtr_t mat2, Real ratio,
        const Eigen::Ref<const Eigen::Matrix<Real, DimM, 1>> & normal_Vector);

    /**
     * This function adds pixels according to the precipitate intersected pixels
     * and the materials involved
     */

    void add_pixels_precipitate(
        std::vector<Ccoord_t<DimS>> intersected_pixels,
        std::vector<Real> intersection_ratios,
        std::vector<Eigen::Matrix<Real, DimM, 1>> intersection_normals,
        MatPtr_t mat1, MatPtr_t mat2);

   protected:
    VectorField_t & normal_vector_field;  //!< field holding the normal vector
                                          //!< of the interface of the layers
    VectorFieldMap_t normal_vector_map;

    ScalarField_t & volume_ratio_field;  //!< field holding the normal vector
    ScalarFieldMap_t volume_ratio_map;

    std::vector<MatPtr_t> material_left_vector{};
    std::vector<MatPtr_t> material_right_vector{};

    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, SplitCell is_cell_split>
    inline void compute_stresses_worker(const StrainField_t & F,
                                        StressField_t & P, TangentField_t & K)
        __attribute__((visibility("default")));
  };

  /* ----------------------------------------------------------------------*/
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto MaterialLaminate<DimS, DimM>::evaluate_stress(s_t && E,
                                                     const size_t & pixel_index,
                                                     Formulation form)
      -> decltype(auto) {
    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;
    auto mat_l = material_left_vector[pixel_index];
    auto mat_r = material_right_vector[pixel_index];
    Strain_t E_eval(E);

    Function_t mat_l_evaluate_stress_tangent_func =
        [&mat_l, pixel_index, form](const Eigen::Ref<const Strain_t> & E) {
          return mat_l->evaluate_stress_tangent_base(std::move(E), pixel_index,
                                                     form);
        };

    Function_t mat_r_evaluate_stress_tangent_func =
        [&mat_r, pixel_index, form](const Eigen::Ref<const Strain_t> & E) {
          return mat_r->evaluate_stress_tangent_base(std::move(E), pixel_index,
                                                     form);
        };

    auto && ratio = this->volume_ratio_map[pixel_index];
    auto && normal_vec = this->normal_vector_map[pixel_index];
    Stress_t ret_stress{};
    switch (form) {
    case Formulation::finite_strain: {
      ret_stress =
          LamHomogen<DimM, Formulation::finite_strain>::evaluate_stress(
              E_eval, mat_l_evaluate_stress_tangent_func,
              mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    case Formulation::small_strain: {
      ret_stress = LamHomogen<DimM, Formulation::small_strain>::evaluate_stress(
          E_eval, mat_l_evaluate_stress_tangent_func,
          mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    default: { std::runtime_error("Unknown formualtion"); }
    }
    return ret_stress;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto MaterialLaminate<DimS, DimM>::evaluate_stress_tangent(
      s_t && E, const size_t & pixel_index, Formulation form)
      -> decltype(auto) {
    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;
    auto && mat_l = material_left_vector[pixel_index];
    auto && mat_r = material_right_vector[pixel_index];
    Strain_t E_eval(E);

    Function_t mat_l_evaluate_stress_tangent_func =
        [&mat_l, pixel_index, form](const Eigen::Ref<const Strain_t> & E) {
          return mat_l->evaluate_stress_tangent_base(std::move(E), pixel_index,
                                                     form);
        };

    Function_t mat_r_evaluate_stress_tangent_func =
        [&mat_r, pixel_index, form](const Eigen::Ref<const Strain_t> & E) {
          return mat_r->evaluate_stress_tangent_base(std::move(E), pixel_index,
                                                     form);
        };
    std::tuple<Stress_t, Stiffness_t> ret_stress_stiffness{};
    auto && ratio = this->volume_ratio_map[pixel_index];
    auto && normal_vec = this->normal_vector_map[pixel_index];
    switch (form) {
    case Formulation::finite_strain: {
      ret_stress_stiffness =
          LamHomogen<DimM, Formulation::finite_strain>::evaluate_stress_tangent(
              E_eval, mat_l_evaluate_stress_tangent_func,
              mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    case Formulation::small_strain: {
      ret_stress_stiffness =
          LamHomogen<DimM, Formulation::small_strain>::evaluate_stress_tangent(
              E_eval, mat_l_evaluate_stress_tangent_func,
              mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
      break;
    }
    default: { std::runtime_error("Unknown formualtion"); }
    }

    return ret_stress_stiffness;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  auto MaterialLaminate<DimS, DimM>::constitutive_law_tangent_small_strain(
      const Eigen::Ref<const Strain_t> & strain, const size_t & pixel_index)
      -> std::tuple<Stress_t, Stiffness_t> {
    Eigen::Map<const Strain_t> F(strain.data());
    return std::move(
        MatTB::constitutive_law_tangent_with_formulation<
            Formulation::small_strain>(*this, std::make_tuple(F), pixel_index));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  auto MaterialLaminate<DimM>::constitutive_law_dynamic(
      const Eigen::Ref<const DynMatrix_t> & strain, const size_t & pixel_index,
      const Formulation & form) -> std::tuple<DynMatrix_t, DynMatrix_t> {
    Eigen::Map < const Eigen::Matrix<Real, DimM, DimM> F(strain.data());

    if (strain.cols() != DimM or strain.rows() != DimM) {
      std::stringstream error {};
      error << "incompatible strain shape, expected " << DimM << " × " << DimM
            << ", but received " << strain.rows() << " × " strain.cols() << ".";
      throw Material(error.str());
    }
    switch (form) {
    case Formulation::finite_strain: {
      return MatTB::constitutive_law_tangent<Formulation::finite_strain>(
            *this, std::make_tuple(F), pixel_index));
      break;
    }
    case Formulation::small_strain: {
      return MatTB::constitutive_law_tangent<Formulation::small_strain>(
            *this, std::make_tuple(F), pixel_index));
      break;
    }
    default:
      throw MaterialError("unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <Formulation Form, SplitCell is_cell_split>
  void MaterialLaminate<DimS, DimM>::compute_stresses_worker(
      const StrainField_t & F, StressField_t & P, TangentField_t & K) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
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
            MatTB::constitutive_law_tangent_with_formulation<Form>(
                *this, strain, index)};
        stress += ratio * std::get<0>(stress_stiffness_mat);
        stiffness += ratio * std::get<1>(stress_stiffness_mat);
      } else {
        stress_stiffness =
            MatTB::constitutive_law_tangent_with_formulation<Form>(
                *this, strain, index);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <Formulation Form, SplitCell is_cell_split>
  void
  MaterialLaminate<DimS, DimM>::compute_stresses_worker(const StrainField_t & F,
                                                        StressField_t & P) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations and
       the infinitesimal strain tensor in small strain problems
    */

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
        stress += ratio * MatTB::constitutive_law_with_formulation<Form>(
                              *this, strain, index);
      } else {
        stress = MatTB::constitutive_law_with_formulation<Form>(*this, strain,
                                                                index);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class... ConstructorArgs>
  std::tuple<std::shared_ptr<MaterialLaminate<DimS, DimM>>,
             MaterialEvaluator<DimM>>
  MaterialLaminate<DimS, DimM>::make_evaluator(ConstructorArgs &&... args) {
    auto mat = std::make_shared<MaterialLaminate<DimS, DimM>>("name", args...);
    using Ret_t = std::tuple<std::shared_ptr<MaterialLaminate<DimS, DimM>>,
                             MaterialEvaluator<DimM>>;
    return Ret_t(mat, MaterialEvaluator<DimM>{mat});
  }
  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre
#endif  // SRC_MATERIALS_MATERIAL_LAMINATE_HH_
