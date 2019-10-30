/**
 * @file   material_stochastic_plasticity.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   24 Jan 2019
 *
 * @brief  material for stochastic plasticity as described in Z. Budrikis et al.
 *         Nature Comm. 8:15928, 2017. It only works together with "python
 *         -script", which performes the avalanche loop. This makes the material
 *         slower but more easy to modify and test.
 *         (copied from material_linear_elastic4.hh)
 *
 * Copyright © 2019 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
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

#ifndef SRC_MATERIALS_MATERIAL_STOCHASTIC_PLASTICITY_HH_
#define SRC_MATERIALS_MATERIAL_STOCHASTIC_PLASTICITY_HH_

#include "common/muSpectre_common.hh"

#include "materials/material_linear_elastic1.hh"
#include "cell/cell_base.hh"
#include "libmugrid/mapped_field.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialStochasticPlasticity;

  /**
   * traits for stochastic plasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialStochasticPlasticity<DimS, DimM>> {
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
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! local field_collections used for internals
    using LFieldColl_t = muGrid::LocalFieldCollection<DimS>;
    //! local Lame constant, plastic increment, stress threshold type
    using ScalarMap_t = muGrid::ScalarFieldMap<LFieldColl_t, Real, true>;

    //! storage type for eigen strain (is updated from outside)
    using EigenStrainMap_t =
        muGrid::MatrixFieldMap<LFieldColl_t, Real, DimM, DimM>;

    //! storage type for an overloaded pixel vector
    using PixelVector_t = std::vector<Ccoord_t<DimS>>;

    //! stochastic plasticity internal variables (Lame 1, Lame 2, eigen strain,
    //! overloaded_pixels)
    using InternalVariables =
      std::tuple<ScalarMap_t, ScalarMap_t, EigenStrainMap_t>;
  };

  /**
   * implements stochastic plasticity with an eigenstrain, Lame constants and
   * plastic flow per pixel.
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialStochasticPlasticity
      : public MaterialMuSpectre<MaterialStochasticPlasticity<DimS, DimM>, DimS,
                                 DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialStochasticPlasticity, DimS, DimM>;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;
    //! global field collection

    //! Full type for stress fields
    using StressField_t =
        muGrid::TensorField<muGrid::GlobalFieldCollection<DimS>, Real,
                            secondOrder, DimM>;

    //! Full type for stress TypedField
    using StressTypedField_t =
        muGrid::TypedField<muGrid::GlobalFieldCollection<DimS>, Real>;

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    using Sys_t = CellBase<DimS, DimM>;

    //! proxy field type for numpy interaction
    using ProxyField_t =
        muGrid::TypedField<typename Sys_t::FieldCollection_t, Real>;

    using Stiffness_t =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    using EigenStrainArg_t = Eigen::Map<Eigen::Matrix<Real, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialStochasticPlasticity>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialStochasticPlasticity() = delete;

    //! Construct by name
    explicit MaterialStochasticPlasticity(std::string name);

    //! Copy constructor
    MaterialStochasticPlasticity(const MaterialStochasticPlasticity & other) =
        delete;

    //! Move constructor
    MaterialStochasticPlasticity(MaterialStochasticPlasticity && other) =
        delete;

    //! Destructor
    virtual ~MaterialStochasticPlasticity() = default;

    //! Copy assignment operator
    MaterialStochasticPlasticity &
    operator=(const MaterialStochasticPlasticity & other) = delete;

    //! Move assignment operator
    MaterialStochasticPlasticity &
    operator=(MaterialStochasticPlasticity && other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor), and the
     * local pixel id.
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E,
                                          const size_t & pixel_index) {
      auto && lambda{this->lambda_field[pixel_index]};
      auto && mu{this->mu_field[pixel_index]};
      auto && eigen_strain{this->eigen_strain_field[pixel_index]};
      return this->evaluate_stress(E, lambda, mu, eigen_strain);
    }

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor), the first
     * Lame constant (lambda) and the second Lame constant (shear modulus/mu).
     */
    template <class s_t>
    inline decltype(auto)
    evaluate_stress(s_t && E, const Real & lambda, const Real & mu,
                    const EigenStrainArg_t & eigen_strain);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor), and the local pixel id.
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t && E,
                                                  const size_t & pixel_index) {
      auto && lambda{this->lambda_field[pixel_index]};
      auto && mu{this->mu_field[pixel_index]};
      auto && eigen_strain{this->eigen_strain_field[pixel_index]};
      return this->evaluate_stress_tangent(E, lambda, mu, eigen_strain);
    }

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor), the first Lame constant (lambda) and
     * the second Lame constant (shear modulus/mu).
     */
    template <class s_t>
    inline decltype(auto)
    evaluate_stress_tangent(s_t && E, const Real & lambda, const Real & mu,
                            const EigenStrainArg_t & eigen_strain);

    /**
     * set the plastic_increment on a single pixel
     **/
    void set_plastic_increment(const Ccoord_t<DimS> pixel,
                               const Real increment);

    /**
     * set the stress_threshold on a single pixel
     **/
    void set_stress_threshold(const Ccoord_t<DimS> pixel, const Real threshold);

    /**
     * set the eigen_strain on a single pixel
     **/
    void set_eigen_strain(
        const Ccoord_t<DimS> pixel,
        Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>> & eigen_strain);

    /**
    * get the plastic_increment on a single pixel
    **/
    const Real & get_plastic_increment(const Ccoord_t<DimS> pixel);

    /**
     * get the stress_threshold on a single pixel
     **/
    const Real & get_stress_threshold(const Ccoord_t<DimS> pixel);

    /**
     * get the eigen_strain on a single pixel
     **/
    const Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>
    get_eigen_strain(const Ccoord_t<DimS> pixel);

    /**
     * reset_overloaded_pixels,
     * reset the internal variable overloaded_pixels by clear the std::vector
     **/
    void reset_overloaded_pixels();

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel, const Real & Youngs_modulus,
                   const Real & Poisson_ratio, const Real & plastic_increment,
                   const Real & stress_threshold,
                   const Eigen::Ref<const Eigen::Matrix<
                       Real, Eigen::Dynamic, Eigen::Dynamic>> & eigen_strain);

    /**
     * evaluate how many pixels have a higher stress than their stress threshold
     */
    inline decltype(auto)
    identify_overloaded_pixels(Sys_t & sys,
                               Eigen::Ref<Vector_t> & stress_numpy_array);

    inline std::vector<Ccoord_t<DimS>> & identify_overloaded_pixels(
        const ProxyField_t & stress_field);

    /**
     * Update the eigen_strain_field of overloaded pixels by a discrete plastic
     * step from the plastic_increment_field in the direction of the deviatoric
     * stress tensor
     */
    inline decltype(auto)
    update_eigen_strain_field(Sys_t & sys,
                              Eigen::Ref<Vector_t> & stress_numpy_array);
    inline void update_eigen_strain_field(const ProxyField_t & stress_field);

    /**
     * Archive the overloaded pixels into an avalanche history
     */
    inline void archive_overloaded_pixels(
        std::list<std::vector<Ccoord_t<DimS>>> & avalanche_history);

    /**
     * relax all overloaded pixels,
     * return the new stress field and the avalance history
     */
    inline decltype(auto)
    relax_overloaded_pixels(Sys_t & sys,
                            Eigen::Ref<Vector_t> & stress_numpy_array);

   protected:
    //! storage for first Lame constant 'lambda',
    //! second Lame constant(shear modulus) 'mu',
    //! plastic strain epsilon_p,
    //! and a vector of overloaded (stress>stress_threshold) pixel
    //! coordinates
    using Field_t =
      muGrid::MappedScalarField<Real, DimS>;
    using LTensor_Field_t =
      muGrid::MappedT2Field<Real, DimS, DimM>;

    Field_t lambda_field;
    Field_t mu_field;
    Field_t plastic_increment_field;
    Field_t stress_threshold_field;
    LTensor_Field_t eigen_strain_field;
    std::vector<Ccoord_t<DimS>> overloaded_pixels;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto MaterialStochasticPlasticity<DimS, DimM>::evaluate_stress(
      s_t && E, const Real & lambda, const Real & mu,
      const EigenStrainArg_t & eigen_strain) -> decltype(auto) {
    return Hooke::evaluate_stress(lambda, mu, E - eigen_strain);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto MaterialStochasticPlasticity<DimS, DimM>::evaluate_stress_tangent(
      s_t && E, const Real & lambda, const Real & mu,
      const EigenStrainArg_t & eigen_strain) -> decltype(auto) {
    muGrid::T4Mat<Real, DimM> C = Hooke::compute_C_T4(lambda, mu);
    return std::make_tuple(
        this->evaluate_stress(std::forward<s_t>(E), lambda, mu, eigen_strain),
        C);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  decltype(auto)  // TypedField<GlobalFieldCollection<DimS>, Real> &
      MaterialStochasticPlasticity<DimS, DimM>::identify_overloaded_pixels(
          Sys_t & sys, Eigen::Ref<Vector_t> & stress_numpy_array) {
    ProxyField_t stress_field{"temp input for stress field",
                              sys.get_collection(), stress_numpy_array,
                              DimM * DimM};
    return this->identify_overloaded_pixels(stress_field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::vector<Ccoord_t<DimS>> &
  MaterialStochasticPlasticity<DimS, DimM>::identify_overloaded_pixels(
      const ProxyField_t & stress_field) {
    auto threshold_map{this->stress_threshold_field.get_map()};
    constexpr bool IsConstMap{true};
    using ProxyMap_t = muGrid::MatrixFieldMap<
      typename ProxyField_t::collection_t, Real, DimM, DimM, IsConstMap>;
    ProxyMap_t stress_map{stress_field};
    std::vector<Ccoord_t<DimS>> & overloaded_pixels_ref{
        this->overloaded_pixels};

    //! loop over all pixels and check if stress overcomes the threshold or not
    for (const auto && pixel_threshold : threshold_map.enumerate()) {
      const auto & pixel{std::get<0>(pixel_threshold)};
      const Real & threshold{std::get<1>(pixel_threshold)};
      const auto & stress{stress_map[pixel]};
      // check if stress is larger than threshold,
      const Real sigma_eq{MatTB::compute_equivalent_von_Mises_stress(stress)};
      // if sigma_eq > threshold write pixel into Ccoord vector
      if ((sigma_eq > threshold) == 1) {  // (sigma_eq > threshold){
        overloaded_pixels_ref.push_back(pixel);
      }
    }
    return overloaded_pixels_ref;
  }

  /* ---------------------------------------------------------------------- */
  //! Updates the eigen strain field of all overloaded pixels by doing a plastic
  //! increment into the deviatoric stress direction by an absolute value given
  //! by the plastic_increment_field
  template <Dim_t DimS, Dim_t DimM>
  decltype(auto)
      MaterialStochasticPlasticity<DimS, DimM>::update_eigen_strain_field(
          Sys_t & sys, Eigen::Ref<Vector_t> & stress_numpy_array) {
    ProxyField_t stress_field {
        "temp input for stress field",
        sys.get_collection(), stress_numpy_array, DimM*DimM};
    return this->update_eigen_strain_field(stress_field);
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  MaterialStochasticPlasticity<DimS, DimM>::update_eigen_strain_field(
      const ProxyField_t & stress_field) {
    constexpr bool IsConstMap{true};
    using ProxyMap_t = muGrid::MatrixFieldMap<
      typename ProxyField_t::collection_t, Real, DimM, DimM, IsConstMap>;
    ProxyMap_t stress_map{stress_field};
    //! initialise strain
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    //! initialise plastic increment
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    //! loop over all overloaded_pixels
    for (const auto & pixel : this->overloaded_pixels) {
      //!  1.) compute plastic_strain_direction = σ_dev/Abs[σ_dev]
      const auto & stress_map_pixel{stress_map[pixel]};
      Eigen::Matrix<Real, DimM, DimM> stress_matrix_pixel = stress_map_pixel;
      const Eigen::Matrix<Real, DimM, DimM> deviatoric_stress{
          MatTB::compute_deviatoric_stress(stress_matrix_pixel)};
      const Real equivalent_stress{
          MatTB::compute_equivalent_von_Mises_stress(stress_map_pixel)};
      const Eigen::Matrix<Real, DimM, DimM> plastic_strain_direction{
          deviatoric_stress / equivalent_stress};

      //!  2.) update eigen_strain_field
      eigen_strain_map[pixel] +=
          plastic_strain_direction * plastic_increment_map[pixel];
    }
  }

  /* ---------------------------------------------------------------------- */
  //! archive_overloaded_pixels(), archives the overloaded pixels saved in
  //! this->overloaded_pixels to the input vector avalanche_history and empties
  //! overloaded_pixels.
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::archive_overloaded_pixels(
      std::list<std::vector<Ccoord_t<DimS>>> & avalanche_history) {
    //!  1.) archive overloaded_pixels in avalanche_history
    avalanche_history.push_back(this->overloaded_pixels);
    //!  2.) clear overloaded pixels
    this->overloaded_pixels.clear();
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_STOCHASTIC_PLASTICITY_HH_
