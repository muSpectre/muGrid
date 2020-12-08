/**
 * @file   material_muSpectre_mechanics.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   16 Jun 2020
 *
 * @brief  abstraction for mechanics materials. Handles the complexities of
 *         stress- and strain conversions, as well as storage of native stress
 *
 * Copyright © 2020 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_MUSPECTRE_MECHANICS_HH_
#define SRC_MATERIALS_MATERIAL_MUSPECTRE_MECHANICS_HH_

#include "material_muSpectre.hh"
namespace muSpectre {

  /**
   * default traits class, should work with most, if not all constitutive laws
   * for solid mechanics
   */
  template <Index_t DimM, StrainMeasure Strain, StressMeasure Stress>
  struct DefaultMechanics_traits {
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
    constexpr static StrainMeasure strain_measure{Strain};
    //! declare what type of stress measure your law yields as output
    constexpr static StressMeasure stress_measure{Stress};

    /**
     * Tensorial rank of this problem and physical units of  input/output fields
     */
    static muGrid::PhysicsDomain get_physics_domain() {
      return muGrid::PhysicsDomain{
          secondOrder, muGrid::Unit::unitless(),
          muGrid::Unit::mass() / (muGrid::Unit::length() *
                                  muGrid::Unit::time() * muGrid::Unit::time())};
    }

    // plain type for strain representation
    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    // plain type for stress representation
    using Stress_t = Strain_t;
    // plain type for tangent representation
    using Tangent_t = muGrid::T4Mat<Real, DimM>;
  };

  template <Index_t DimM, StrainMeasure Strain, StressMeasure Stress>
  constexpr StrainMeasure
      DefaultMechanics_traits<DimM, Strain, Stress>::strain_measure;
  template <Index_t DimM, StrainMeasure Strain, StressMeasure Stress>
  constexpr StressMeasure
      DefaultMechanics_traits<DimM, Strain, Stress>::stress_measure;

  /**
   * Base class for most convenient implementation of constitutive laws solid
   * mechanics
   */
  template <class Material, Index_t DimM>
  class MaterialMuSpectreMechanics
      : public MaterialMuSpectre<Material, DimM, MaterialMechanicsBase> {
   public:
    using Parent = MaterialMuSpectre<Material, DimM, MaterialMechanicsBase>;

    //! traits for the CRTP subclass
    using traits = MaterialMuSpectre_traits<Material>;
    using DynMatrix_t = typename Parent::DynMatrix_t;

    using Strain_t = typename Parent::Strain_t;
    using Stress_t = typename Parent::Stress_t;
    using Tangent_t = typename Parent::Tangent_t;

    // just use MaterialMuSpectre's constructors, this class is interface-only
    using Parent::Parent;

    //! Default constructor
    MaterialMuSpectreMechanics() = delete;

    //! Copy constructor
    MaterialMuSpectreMechanics(const MaterialMuSpectreMechanics & other) =
        delete;

    //! Move constructor
    MaterialMuSpectreMechanics(MaterialMuSpectreMechanics && other) = delete;

    //! Destructor
    virtual ~MaterialMuSpectreMechanics() = default;

    //! Copy assignment operator
    MaterialMuSpectreMechanics &
    operator=(const MaterialMuSpectreMechanics & other) = delete;

    //! Move assignment operator
    MaterialMuSpectreMechanics &
    operator=(MaterialMuSpectreMechanics && other) = delete;

    //! computes stress
    using Parent::compute_stresses;
    using Parent::compute_stresses_tangent;

    //! computes stress
    inline void
    compute_stresses(const muGrid::RealField & F, muGrid::RealField & P,
                     const SplitCell & is_cell_split = SplitCell::no,
                     const StoreNativeStress & store_native_stress =
                         StoreNativeStress::no) final;

    //! computes stress and tangent modulus
    inline void
    compute_stresses_tangent(const muGrid::RealField & F, muGrid::RealField & P,
                             muGrid::RealField & K,
                             const SplitCell & is_cell_split = SplitCell::no,
                             const StoreNativeStress & store_native_stress =
                                 StoreNativeStress::no) final;

    inline std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & pixel_index) final;

    /**
     * returns a reference to the currently set formulation
     */
    const Formulation & get_formulation() const final { return this->form; }

    /**
     * set the formulation
     */
    void set_formulation(const Formulation & form) override {
      this->form = form;
    }

    //! checks whether this material can be used in small strain formulation
    inline static void check_small_strain_capability();

    /**
     * returns the expected strain measure of the material
     */
    StrainMeasure get_expected_strain_measure() const final {
      return traits::strain_measure;
    }

    using Parent::make;
    //! Factory. The ConstructorArgs refer the arguments after `name`
    template <class... ConstructorArgs>
    inline static Material & make(std::shared_ptr<Cell> cell,
                                  const std::string & name,
                                  ConstructorArgs &&... args);

   protected:
    //! dispatches the correct formulation to the next dispatcher
    template <Formulation Form, SplitCell IsSplit, class... Args>
    inline void
    compute_stresses_dispatch1(const StoreNativeStress store_native_stress,
                               Args &&... args);

    //! dispatches the correct solver type to the next dispatcher
    template <Formulation Form, StrainMeasure StoredStrain, SplitCell IsSplit,
              class... Args>
    inline void
    compute_stresses_dispatch2(const StoreNativeStress store_native_stress,
                               Args &&... args);

    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, StrainMeasure StoredStrain,
              SplitCell is_cell_split, StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P)
        __attribute__((visibility("default")));

    //! computes stress with the formulation available at compile time as well
    //! as the info whether to store native stresses
    //! __attribute__ required by g++-6 and g++-7 because of this bug:
    //! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
    template <Formulation Form, StrainMeasure StoredStrain,
              SplitCell is_cell_split, StoreNativeStress DoStoreNative>
    inline void compute_stresses_worker(const muGrid::RealField & F,
                                        muGrid::RealField & P,
                                        muGrid::RealField & K)
        __attribute__((visibility("default")));

    Formulation form{Formulation::not_set};
  };
  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <Formulation Form, StrainMeasure StoredStrain, SplitCell IsSplit,
            class... Args>
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses_dispatch2(
      const StoreNativeStress store_native_stress, Args &&... args) {
    switch (store_native_stress) {
    case StoreNativeStress::yes: {
      this->template compute_stresses_worker<Form, StoredStrain, IsSplit,
                                             StoreNativeStress::yes>(
          std::forward<Args>(args)...);
      break;
    }
    case StoreNativeStress::no: {
      this->template compute_stresses_worker<Form, StoredStrain, IsSplit,
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
  template <Formulation Form, SplitCell IsSplit, class... Args>
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses_dispatch1(
      const StoreNativeStress store_native_stress, Args &&... args) {
    switch (this->get_solver_type()) {
    case SolverType::FiniteElements: {
      constexpr auto StoredStrain{
          Form == Formulation::native
              ? traits::strain_measure
              : get_stored_strain_type(Form, SolverType::FiniteElements)};

      this->template compute_stresses_dispatch2<Form, StoredStrain, IsSplit>(
          store_native_stress, std::forward<Args>(args)...);
      break;
    }
    case SolverType::Spectral: {
      constexpr auto StoredStrain{
          Form == Formulation::native
              ? traits::strain_measure
              : get_stored_strain_type(Form, SolverType::Spectral)};

      this->template compute_stresses_dispatch2<Form, StoredStrain, IsSplit>(
          store_native_stress, std::forward<Args>(args)...);
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown value for store native stress");
      break;
    }
  }

  namespace MatTB {
    /* ----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strain, class Stress, class Op, class NativeTreat>
    void evaluate_material_stress(Material & mat, Strain && strain,
                                  Stress & stress, const size_t & quad_pt_id,
                                  const Op & operation,
                                  NativeTreat & native_stress_treatment) {
      using traits = MaterialMuSpectre_traits<Material>;

      constexpr StrainMeasure expected_strain_m{
          get_formulation_strain_type(Form, traits::strain_measure)};

      switch (Form) {
      case Formulation::small_strain: {
        auto && eps{MatTB::convert_strain<StoredStrain, expected_strain_m>(
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
        auto && E{MatTB::convert_strain<StoredStrain, expected_strain_m>(grad)};
        auto && stress_result{mat.evaluate_stress(std::move(E), quad_pt_id)};

        // the following is a no-op if store_native_stress in not 'yes'
        native_stress_treatment(stress_result);
        // if this function gets built with any strain other that H or F as
        // input, it means it wont ever be executed, so we can ignore it at
        // compile time and yell at runtime
        if ((StoredStrain != StrainMeasure::PlacementGradient) and
            (StoredStrain != StrainMeasure::DisplacementGradient)) {
          throw MaterialError(
              "This function handles finite strain and should only ever be "
              "called with the placement gradient F or the displacement "
              "gradient H as input");
        }
        constexpr StrainMeasure InputStrain{
            StoredStrain == StrainMeasure::PlacementGradient
                ? StrainMeasure::PlacementGradient
                : StrainMeasure::DisplacementGradient};

        operation(
            ::muSpectre::MatTB::PK1_stress<traits::stress_measure,
                                           traits::strain_measure>(
                MatTB::convert_strain<InputStrain,
                                      StrainMeasure::PlacementGradient>(grad),
                std::move(stress_result)),
            stress);
        break;
      }
      case Formulation::native: {
        auto && strain_converted{
            MatTB::convert_strain<StoredStrain, expected_strain_m>(
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
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses>
    void constitutive_law(Material & mat, Strains && strains,
                          Stresses & stresses, const size_t & quad_pt_id,
                          const Real & ratio) {
      OperationAddition operation_addition(ratio);
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_addition,
          stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses>
    void constitutive_law(Material & mat, Strains && strains,
                          Stresses & stresses, const size_t & quad_pt_id) {
      OperationAssignment operation_assignment;
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_assignment,
          stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses,
              Dim_t Dim>  // Dim_t is used on purpose, as it is inferred
    void constitutive_law(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id, const Real & ratio,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      static_assert(Dim == Material::MaterialDimension(),
                    "Dim is a SFINAE parameter, do not set it");
      OperationAddition operation_addition(ratio);
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_addition,
          stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses>
    void constitutive_law(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id,
        Eigen::Map<Eigen::Matrix<Real, Material::MaterialDimension(),
                                 Material::MaterialDimension()>> &
            native_stress) {
      constexpr Index_t Dim{Material::MaterialDimension()};
      OperationAssignment operation_assignment;
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_assignment,
          stress_treatment);
    }

    /* ---------------------------------------------------------------------- */
    template <StrainMeasure StoredStrain, class Material, class Strain,
              class Stress, class Stiffness, class Op, class NativeTreat>
    void evaluate_material_stress_tangent_finite_strain(
        Material & mat, Strain && strain,
        std::tuple<Stress, Stiffness> & stress_stiffness,
        const size_t & quad_pt_id, const Op & operation,
        NativeTreat & native_stress_treatment) {
      using traits = MaterialMuSpectre_traits<Material>;

      constexpr StrainMeasure expected_strain_m{get_formulation_strain_type(
          Formulation::finite_strain, traits::strain_measure)};
      auto && grad{std::get<0>(strain)};
      // if this function gets built with any strain other that H or F as
      // input, it means it wont ever be executed, so we can ignore it at
      // compile time and yell at runtime
      if ((StoredStrain != StrainMeasure::PlacementGradient) and
          (StoredStrain != StrainMeasure::DisplacementGradient)) {
        throw MaterialError(
            "This function handles finite strain and should only ever be "
            "called with the placement gradient F or the displacement "
            "gradient H as input");
      }
      constexpr StrainMeasure InputStrain{
          StoredStrain == StrainMeasure::PlacementGradient
              ? StrainMeasure::PlacementGradient
              : StrainMeasure::DisplacementGradient};
      auto && E{MatTB::convert_strain<InputStrain, expected_strain_m>(grad)};
      auto && stress_stiffness_mat{
          mat.evaluate_stress_tangent(std::move(E), quad_pt_id)};

      // the following is a no-op if store_native_stress in not 'yes'
      native_stress_treatment(std::get<0>(stress_stiffness_mat));
      auto && stress_stiffness_mat_converted{::muSpectre::MatTB::PK1_stress<
          traits::stress_measure, traits::strain_measure>(
          MatTB::convert_strain<InputStrain, StrainMeasure::PlacementGradient>(
              grad),
          std::move(std::get<0>(stress_stiffness_mat)),
          std::move(std::get<1>(stress_stiffness_mat)))};

      operation(std::get<0>(stress_stiffness_mat_converted),
                std::get<0>(stress_stiffness));
      operation(std::get<1>(stress_stiffness_mat_converted),
                std::get<1>(stress_stiffness));
    }
    /* ---------------------------------------------------------------------- */
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strain, class Stress, class Stiffness, class Op,
              class NativeTreat>
    void evaluate_material_stress_tangent(
        Material & mat, Strain && strain,
        std::tuple<Stress, Stiffness> & stress_stiffness,
        const size_t & quad_pt_id, const Op & operation,
        NativeTreat & native_stress_treatment) {
      using traits = MaterialMuSpectre_traits<Material>;

      constexpr StrainMeasure expected_strain_m{
          get_formulation_strain_type(Form, traits::strain_measure)};
      switch (Form) {
      case Formulation::small_strain: {
        auto && eps{MatTB::convert_strain<StoredStrain, expected_strain_m>(
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
        evaluate_material_stress_tangent_finite_strain<StoredStrain>(
            mat, std::forward<Strain>(strain), stress_stiffness, quad_pt_id,
            operation, native_stress_treatment);
        break;
      }
      case Formulation::native: {
        auto && strain_converted{
            MatTB::convert_strain<StoredStrain, expected_strain_m>(
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
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses>
    void constitutive_law_tangent(Material & mat, Strains && strains,
                                  Stresses & stresses,
                                  const size_t & quad_pt_id) {
      OperationAssignment operation_assignment{};
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress_tangent<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_assignment,
          stress_treatment);
    }

    /* ----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses,
              Dim_t Dim>  // Dim_t is used on purpose, as it is inferred
    void constitutive_law_tangent(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      static_assert(Dim == Material::MaterialDimension(),
                    "Dim is a SFINAE parameter, do not touch it");
      OperationAssignment operation_assignment{};
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress_tangent<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_assignment,
          stress_treatment);
    }

    /*----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses>
    void constitutive_law_tangent(Material & mat, Strains && strains,
                                  Stresses & stresses,
                                  const size_t & quad_pt_id,
                                  const Real & ratio) {
      OperationAddition operation_addition{ratio};
      NativeStressTreatment<StoreNativeStress::no> stress_treatment{};
      evaluate_material_stress_tangent<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_addition,
          stress_treatment);
    }

    /*----------------------------------------------------------------------*/
    template <Formulation Form, StrainMeasure StoredStrain, class Material,
              class Strains, class Stresses,
              Dim_t Dim>  // Dim_t is used on purpose, as it is inferred
    void constitutive_law_tangent(
        Material & mat, Strains && strains, Stresses & stresses,
        const size_t & quad_pt_id, const Real & ratio,
        Eigen::Map<Eigen::Matrix<Real, Dim, Dim>> & native_stress) {
      static_assert(Dim == Material::MaterialDimension(),
                    "Dim is a SFINAE parameter, do not touch it");
      OperationAddition operation_addition{ratio};
      NativeStressTreatment<StoreNativeStress::yes, Dim> stress_treatment{
          native_stress};
      evaluate_material_stress_tangent<Form, StoredStrain>(
          mat, strains, stresses, quad_pt_id, operation_addition,
          stress_treatment);
    }
  }  // namespace MatTB

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses(
      const muGrid::RealField & F, muGrid::RealField & P,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    switch (this->get_formulation()) {
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
      this->check_small_strain_capability();
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
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses_tangent(
      const muGrid::RealField & F, muGrid::RealField & P, muGrid::RealField & K,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    switch (this->get_formulation()) {
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
      this->check_small_strain_capability();
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
  template <Formulation Form, StrainMeasure StoredStrain, SplitCell IsCellSplit,
            StoreNativeStress DoStoreNative>
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P,
      muGrid::RealField & K) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems

       The internal_variables tuple contains whatever internal variables
       Material declared (e.g., eigenstrain, strain rate, etc.)
    */
    auto & this_mat{static_cast<Material &>(*this)};
    if (Form == Formulation::small_strain) {
      this->check_small_strain_capability();
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
          MatTB::constitutive_law_tangent<Form, StoredStrain>(
              this_mat, strain, stress_stiffness, quad_pt_id, ratio);
        } else {
          MatTB::constitutive_law_tangent<Form, StoredStrain>(
              this_mat, strain, stress_stiffness, quad_pt_id);
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
          MatTB::constitutive_law_tangent<Form, StoredStrain>(
              this_mat, strain, stress_stiffness, quad_pt_id, ratio,
              quad_pt_native_stress);
        } else {
          MatTB::constitutive_law_tangent<Form, StoredStrain>(
              this_mat, strain, stress_stiffness, quad_pt_id,
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
  template <Formulation Form, StrainMeasure StoredStrain, SplitCell IsCellSplit,
            StoreNativeStress DoStoreNative>
  void MaterialMuSpectreMechanics<Material, DimM>::compute_stresses_worker(
      const muGrid::RealField & F, muGrid::RealField & P) {
    /* These lambdas are executed for every integration point.

       F contains the transformation gradient for finite strain calculations
       and the infinitesimal strain tensor in small strain problems
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
          MatTB::constitutive_law<Form, StoredStrain>(this_mat, strain, stress,
                                                      quad_pt_id, ratio);
        } else {
          MatTB::constitutive_law<Form, StoredStrain>(this_mat, strain, stress,
                                                      quad_pt_id);
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
          MatTB::constitutive_law<Form, StoredStrain>(this_mat, strain, stress,
                                                      quad_pt_id, ratio,
                                                      quad_pt_native_stress);
        } else {
          MatTB::constitutive_law<Form, StoredStrain>(
              this_mat, strain, stress, quad_pt_id, quad_pt_native_stress);
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
  auto MaterialMuSpectreMechanics<Material, DimM>::constitutive_law_dynamic(
      const Eigen::Ref<const DynMatrix_t> & strain,
      const size_t & quad_pt_index) -> std::tuple<DynMatrix_t, DynMatrix_t> {
    auto & this_mat = static_cast<Material &>(*this);
    Eigen::Map<const Strain_t> F(strain.data());

    std::tuple<Stress_t, Tangent_t> PK{};

    if (strain.cols() != DimM or strain.rows() != DimM) {
      std::stringstream error{};
      error << "incompatible strain shape, expected " << DimM << " × " << DimM
            << ", but received " << strain.rows() << " × " << strain.cols()
            << "." << std::endl;
      throw MaterialError(error.str());
    }

    switch (this->get_formulation()) {
    case Formulation::finite_strain: {
      switch (this->get_solver_type()) {
      case SolverType::FiniteElements: {
        constexpr auto StoredStrain{get_stored_strain_type(
            Formulation::finite_strain, SolverType::FiniteElements)};

        MatTB::constitutive_law_tangent<Formulation::finite_strain,
                                        StoredStrain>(
            this_mat, std::make_tuple(F), PK, quad_pt_index);

        break;
      }
      case SolverType::Spectral: {
        constexpr auto StoredStrain{get_stored_strain_type(
            Formulation::finite_strain, SolverType::Spectral)};

        MatTB::constitutive_law_tangent<Formulation::finite_strain,
                                        StoredStrain>(
            this_mat, std::make_tuple(F), PK, quad_pt_index);

        break;
      }
      default:
        throw MaterialError("Unknown solver type");
        break;
      }
      break;
    }
    case Formulation::small_strain: {
      switch (this->get_solver_type()) {
      case SolverType::FiniteElements: {
        constexpr auto StoredStrain{get_stored_strain_type(
            Formulation::small_strain, SolverType::FiniteElements)};

        MatTB::constitutive_law_tangent<Formulation::small_strain,
                                        StoredStrain>(
            this_mat, std::make_tuple(F), PK, quad_pt_index);

        break;
      }
      case SolverType::Spectral: {
        constexpr auto StoredStrain{get_stored_strain_type(
            Formulation::small_strain, SolverType::Spectral)};

        MatTB::constitutive_law_tangent<Formulation::small_strain,
                                        StoredStrain>(
            this_mat, std::make_tuple(F), PK, quad_pt_index);

        break;
      }
      default:
        throw MaterialError("Unknown solver type");
        break;
      }
      break;
    }
    default:
      throw MaterialError("Unknown formulation");
      break;
    }
    return PK;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  template <class... ConstructorArgs>
  Material &
  MaterialMuSpectreMechanics<Material, DimM>::make(std::shared_ptr<Cell> cell,
                                                   const std::string & name,
                                                   ConstructorArgs &&... args) {
    auto mat{std::make_unique<Material>(name, cell->get_spatial_dim(),
                                        cell->get_nb_quad_pts(), args...)};
    auto && form{cell->get_formulation()};
    if (form == Formulation::small_strain) {
      Material::check_small_strain_capability();
    }

    auto & mat_ref{*mat};
    auto is_cell_split{cell->get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell->add_material(std::move(mat));
    return mat_ref;
  }

  /* ---------------------------------------------------------------------- */
  template <class Material, Index_t DimM>
  void
  MaterialMuSpectreMechanics<Material, DimM>::check_small_strain_capability() {
    if (not(is_objective(traits::strain_measure))) {
      std::stringstream err_str{};
      err_str
          << "The material expected strain measure is: "
          << traits::strain_measure
          << ", while in small strain the required strain measure should be "
             "objective (in order to be obtainable from infinitesimal "
             "strain)."
          << " Accordingly, this material is not meant to be utilized in "
             "small strain formulation"
          << std::endl;
      throw(muGrid::RuntimeError(err_str.str()));
    }
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_MUSPECTRE_MECHANICS_HH_
