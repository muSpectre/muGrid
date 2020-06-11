/**
 * @file   s_t_material_linear_elastic_generic1.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Jan 2020
 *
 * @brief  Material that is merely used to behave as an intermediate convertor
 * for enablling us to conduct tests on stress_transformation usogn
 * MaterialLinearelasticgeneric1
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

#ifndef SRC_MATERIALS_S_T_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_
#define SRC_MATERIALS_S_T_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_

#include "materials/material_linear_elastic_generic1.hh"
#include "materials/stress_transformations_PK1.hh"
#include "materials/stress_transformations_Kirchhoff.hh"

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  class STMaterialLinearElasticGeneric1;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */

  template <Index_t DimM, StrainMeasure StrainMIn, StressMeasure StressMOut>
  struct MaterialMuSpectre_traits<
      STMaterialLinearElasticGeneric1<DimM, StrainMIn, StressMOut>> {
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
    constexpr static auto strain_measure{StrainMIn};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMOut};
  };

  /**
   * Linear elastic law defined by a full stiffness tensor with the ability to
   * compile and work for different strain/stress measures
   */
  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  class STMaterialLinearElasticGeneric1
      : public MaterialMuSpectre<
            STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>, DimM> {
   public:
    //! base class:
    using Parent = MaterialMuSpectre<
        STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>, DimM>;

    using CInput_t = Eigen::Ref<Eigen::MatrixXd>;

    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stress_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<
        STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>>;

    //! Default constructor
    STMaterialLinearElasticGeneric1() = delete;

    /**
     * Constructor by name and stiffness tensor.
     *
     * @param name unique material name
     * @param spatial_dimension spatial dimension of the problem. This
     * corresponds to the dimensionality of the Cell
     * @param nb_quad_pts number of quadrature points per pixel
     * @param C_voigt elastic tensor in Voigt notation
     */
    STMaterialLinearElasticGeneric1(const std::string & name,
                                    const Index_t & spatial_dimension,
                                    const Index_t & nb_quad_pts,
                                    const CInput_t & C_voigt);

    //! Copy constructor
    STMaterialLinearElasticGeneric1(
        const STMaterialLinearElasticGeneric1 & other) = delete;

    //! Move constructor
    STMaterialLinearElasticGeneric1(STMaterialLinearElasticGeneric1 && other) =
        default;

    //! Destructor
    virtual ~STMaterialLinearElasticGeneric1() = default;

    //! Copy assignment operator
    STMaterialLinearElasticGeneric1 &
    operator=(const STMaterialLinearElasticGeneric1 & other) = delete;

    //! Move assignment operator
    STMaterialLinearElasticGeneric1 &
    operator=(STMaterialLinearElasticGeneric1 && other) = delete;

    using Material_sptr = std::shared_ptr<STMaterialLinearElasticGeneric1>;

    //! Factory
    static std::tuple<Material_sptr, MaterialEvaluator<DimM>>
    make_evaluator(const CInput_t & C_voigt);

    /**
     * evaluates stress given the strain
     */
    template <class Derived>
    inline Stress_t evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                    const size_t & quad_pt_index = 0);

    /**
     * evaluates both stress and stiffness given the strain
     */
    template <class Derived>
    inline std::tuple<Stress_t, Stiffness_t>
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & strain,
                            const size_t & quad_pt_index = 0);

    inline void set_F(const Strain_t & Finp) {
      this->F = Finp;
      this->F_is_set = true;
    }

    Stiffness_t get_C() { return this->C; }

   protected:
    // Here, the stiffness tensor is encapsulated into a unique ptr because
    // of this bug:
    // https://eigen.tuxfamily.narkive.com/maHiFSha/fixed-size-vectorizable-members-and-std-make-shared
    // . The problem is that `std::make_shared` uses the global `::new` to
    // allocate `void *` rather than using the the object's `new` operator,
    // and therefore ignores the solution proposed by eigen (documented here
    // http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html).
    // Offloading the offending object into a heap-allocated structure who's
    // construction we control fixes this problem temporarily, until we can
    // use C++17 and guarantee alignment. This comes at the cost of a heap
    // allocation, which is not an issue here, as this happens only once per
    // material and run.
    std::unique_ptr<Stiffness_t> C_holder;  //! stiffness
                                            //! tensor
    const Stiffness_t & C;

    // The Gradient that is needed to carry out the stress_transformations in
    // evaluate_stess() function
    std::unique_ptr<Strain_t> F_holder;
    Strain_t & F;
    bool F_is_set;
  };

  /* ---------------------------------------------------------------------- */

  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  template <class Derived>
  auto STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & strain, const size_t &
      /*quad_pt_index*/) -> Stress_t {
    static_assert(Derived::ColsAtCompileTime == DimM, "wrong input size");
    static_assert(Derived::RowsAtCompileTime == DimM, "wrong input size");

    // Be careful that this F should be compatible with the strain that is
    // passed to material in whenever the evaluate_stress function is called.
    if (not this->F_is_set) {
      throw(muGrid::RuntimeError(
          "The gradient should be set using set_F(F), otherwise you are not "
          "allowed to use this function (it is nedded for "
          "stress_transformation)"));
    }

    // We have to convert strain to Green-Lagrange before passing it to the
    // Parent material stress_evaluate function which is a
    // MaterialLinearElasticGeneric1
    Strain_t E{
        MatTB::convert_strain<StrainM, StrainMeasure::GreenLagrange>(strain)};

    // S is the returned stress similar to  evaluate_stress function of
    // MaterialLinearElasticGeneric1 which is stress in PK2 measure
    // i.e.: S = C * E
    Stress_t S{Matrices::tensmult(this->C, E)};
    Strain_t F_input{Strain_t::Zero()};

    if (StrainM == StrainMeasure::Gradient) {
      F_input = strain;
    } else {
      F_input = this->F;
    }
    switch (StressM) {
    case StressMeasure::PK2: {
      return S;
    }
    case StressMeasure::PK1: {
      Stress_t ret_stress{
          MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
              std::move(F), std::move(S))
              .eval()};
      return ret_stress;
    }
    case StressMeasure::Kirchhoff: {
      Stress_t ret_stress{MatTB::Kirchhoff_stress<StressMeasure::PK2,
                                                  StrainMeasure::GreenLagrange>(
                              std::move(F_input), std::move(S))
                              .eval()};
      return ret_stress;
    }
    default: {
      std::stringstream err{};
      err << "The stress transforamtion needed to return the desired stress "
             "measure ("
          << StressM
          << ") is not defined. Please make sure that the stress "
             "transforamtion from PK2 to"
          << StressM << " is implemented.";
      throw(muGrid::RuntimeError(err.str()));
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM, StrainMeasure StrainM, StressMeasure StressM>
  template <class Derived>
  auto STMaterialLinearElasticGeneric1<DimM, StrainM, StressM>::
      evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & strain,
                              const size_t &
                              /*quad_pt_index*/)
          -> std::tuple<Stress_t, Stiffness_t> {
    std::stringstream err{};
    err << "You are not allowed to use this function beacuse this material is "
           "implemented to be used merely through "
           "MaterialEvaluator<DimM>::estimate_tangent "
           "which is supposedly needless of this function and just needs the "
           "evaluate_stress(...) function. However, if once it became "
           "necessary to use this function it is necessary first to implement "
           "the conversion of PK2 to all required stress_tangent measures."
        << std::endl;
    throw(muGrid::RuntimeError(err.str()));
    using Stiffness_t = Eigen::Map<const muGrid::T4Mat<Real, DimM>>;
    using Ret_t = std::tuple<Stress_t, Stiffness_t>;
    // This return statment is here only to make the declared auto return type
    // compatible with other materials. Therefore, the compilation will be
    // carried out. However, it actually never manages to retrun because of the
    // above error that is throwed in runtime.
    return Ret_t{this->evaluate_stress(strain), Stiffness_t(this->C.data())};
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_S_T_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_
