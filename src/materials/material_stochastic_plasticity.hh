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
#include "cell/cell.hh"

#include <libmugrid/mapped_field.hh>
#include <libmugrid/field_map_static.hh>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialStochasticPlasticity;

  /**
   * traits for stochastic plasticity with eigenstrain
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialStochasticPlasticity<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * implements stochastic plasticity with an eigenstrain, Lameconstants and
   * plastic flow per pixel.
   */
  template <Index_t DimM>
  class MaterialStochasticPlasticity
      : public MaterialMuSpectreMechanics<MaterialStochasticPlasticity<DimM>,
                                          DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectreMechanics<MaterialStochasticPlasticity, DimM>;

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

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
    explicit MaterialStochasticPlasticity(const std::string & name,
                                          const Index_t & spatial_dimension,
                                          const Index_t & nb_quad_pts);

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
     * set the plastic_increment on a single quadrature point
     **/
    void set_plastic_increment(const size_t & quad_pt_id,
                               const Real & increment);

    /**
     * set the stress_threshold on a single quadrature point
     **/
    void set_stress_threshold(const size_t & quad_pt_id,
                              const Real & threshold);

    /**
     * set the eigen_strain on a single quadrature point
     **/
    void set_eigen_strain(
        const size_t & quad_pt_id,
        Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>> & eigen_strain);

    /**
     * get the plastic_increment on a single quadrature point
     **/
    const Real & get_plastic_increment(const size_t & quad_pt_id);

    /**
     * get the stress_threshold on a single quadrature point
     **/
    const Real & get_stress_threshold(const size_t & quad_pt_id);

    /**
     * get the eigen_strain on a single quadrature point
     **/
    const Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>
    get_eigen_strain(const size_t & quad_pt_id);

    /**
     * reset_overloaded_quadrature points,
     * reset the internal variable overloaded_quad_pts by clear the std::vector
     **/
    void reset_overloaded_quad_pts();

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const size_t & pixel_id) final;

    /**
     * overload add_pixel all material parameters are defined per pixel thus if
     * a pixel has several quad points the material parameters on all quad
     * points belonging to a pixel are equal.
     */
    void add_pixel(const size_t & pixel_id, const Real & Youngs_modulus,
                   const Real & Poisson_ratio, const Real & plastic_increment,
                   const Real & stress_threshold,
                   const Eigen::Ref<const Eigen::Matrix<
                       Real, Eigen::Dynamic, Eigen::Dynamic>> & eigen_strain);

    /**
     * overload add_pixel Youngs_modulus and Poisson_ratio are defined per
     * pixel, plastic_increment, stress_threshold and eigen_strain are defined
     * per quad point. Therefore you have to hand over Eigen matrices
     * containing in each row the value(s) for one quad point.
     */
    void add_pixel(
        const size_t & pixel_id, const Real & Youngs_modulus,
        const Real & Poisson_ratio,
        const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
            plastic_increment,
        const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
            stress_threshold,
        const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic,
                                             Eigen::Dynamic>> & eigen_strain);

    /**
     * evaluate how many pixels have a higher stress than their stress threshold
     */
    inline decltype(auto)
    identify_overloaded_quad_pts(Cell & cell);

    inline decltype(auto)
    identify_overloaded_quad_pts(Cell & cell,
                                 Eigen::Ref<Vector_t> & stress_numpy_array);

    inline std::vector<size_t> & identify_overloaded_quad_pts(
        const muGrid::TypedFieldBase<Real> & stress_field,
        const size_t & local_quad_pt_id_offset);

    /**
     * Update the eigen_strain_field of overloaded pixels by a discrete plastic
     * step from the plastic_increment_field in the direction of the deviatoric
     * stress tensor
     */
    inline decltype(auto)
    update_eigen_strain_field(Cell & cell,
                              Eigen::Ref<Vector_t> & stress_numpy_array);

    /* ---------------------------------------------------------------------- */
    inline void update_eigen_strain_field(
        const muGrid::TypedFieldBase<Real> & stress_field);

    /**
     * Archive the overloaded pixels into an avalanche history
     */
    inline void archive_overloaded_quad_pts(
        std::list<std::vector<size_t>> & avalanche_history);

    /**
     * relax all overloaded pixels,
     * return the new stress field and the avalance history
     */
    inline decltype(auto)
    relax_overloaded_quad_pts(Cell & cell,
                              Eigen::Ref<Vector_t> & stress_numpy_array);

   protected:
    //! storage for first Lame constant 'lambda',
    //! second Lame constant(shear modulus) 'mu',
    //! plastic strain epsilon_p,
    //! and a vector of overloaded (stress>stress_threshold) pixel coordinates
    using Field_t =
        muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt>;
    using LTensor_Field_t =
        muGrid::MappedT2Field<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    Field_t lambda_field;
    Field_t mu_field;
    Field_t plastic_increment_field;
    Field_t stress_threshold_field;
    LTensor_Field_t eigen_strain_field;
    std::vector<size_t> overloaded_quad_pts{};
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialStochasticPlasticity<DimM>::evaluate_stress(
      s_t && E, const Real & lambda, const Real & mu,
      const EigenStrainArg_t & eigen_strain) -> decltype(auto) {
    return Hooke::evaluate_stress(lambda, mu, E - eigen_strain);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialStochasticPlasticity<DimM>::evaluate_stress_tangent(
      s_t && E, const Real & lambda, const Real & mu,
      const EigenStrainArg_t & eigen_strain) -> decltype(auto) {
    muGrid::T4Mat<Real, DimM> C = Hooke::compute_C_T4(lambda, mu);
    return std::make_tuple(
        this->evaluate_stress(std::forward<s_t>(E), lambda, mu, eigen_strain),
        C);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  decltype(auto)
  MaterialStochasticPlasticity<DimM>::identify_overloaded_quad_pts(
      Cell & cell) {
    // check if the native stress was stored
    if (not this->has_native_stress()) {
      throw MaterialError(
          "The native stress was not stored. Either use one of the "
          "'identify_overloaded_quad_pts' that takes the stress field as "
          "parameter or turn StoreNativeStress on.");
    }

    auto && PK2_stress_field{this->get_native_stress()};

    // compute quad point offset for local quad point ids
    const DynCcoord_t & subdomain_locs{
        cell.get_projection().get_subdomain_locations()};
    const DynCcoord_t & nb_domain_grid_pts{
        cell.get_projection().get_nb_domain_grid_pts()};
    size_t local_quad_pt_id_offset{0};
    int dim{subdomain_locs.get_dim()};
    size_t factor{static_cast<size_t>(cell.get_nb_quad_pts())};
    for (int i = 0; i < dim; i++) {
      local_quad_pt_id_offset += subdomain_locs[i] * factor;
      if (i != dim - 1) {
          factor *= nb_domain_grid_pts[i];
      }
    }

    return this->identify_overloaded_quad_pts(PK2_stress_field,
                                              local_quad_pt_id_offset);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  decltype(auto)
      MaterialStochasticPlasticity<DimM>::identify_overloaded_quad_pts(
          Cell & cell, Eigen::Ref<Vector_t> & stress_numpy_array) {
    muGrid::WrappedField<Real> stress_field{
        "temp input for stress field", cell.get_fields(), DimM * DimM,
        stress_numpy_array, QuadPtTag};

    // compute quad point offset for local quad point ids
    const DynCcoord_t & subdomain_locs{
        cell.get_projection().get_subdomain_locations()};
    const DynCcoord_t & nb_domain_grid_pts{
        cell.get_projection().get_nb_domain_grid_pts()};
    size_t local_quad_pt_id_offset{0};
    int dim{subdomain_locs.get_dim()};
    size_t factor{static_cast<size_t>(cell.get_nb_quad_pts())};
    for (int i = 0; i < dim; i++) {
      local_quad_pt_id_offset += subdomain_locs[i] * factor;
      if (i != dim - 1) {
          factor *= nb_domain_grid_pts[i];
      }
    }
    return this->identify_overloaded_quad_pts(stress_field,
                                              local_quad_pt_id_offset);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  std::vector<size_t> &
  MaterialStochasticPlasticity<DimM>::identify_overloaded_quad_pts(
      const muGrid::TypedFieldBase<Real> & stress_field,
      const size_t & local_quad_pt_id_offset) {
    auto & threshold_map{this->stress_threshold_field.get_map()};

    muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>
        stress_map{stress_field};
    std::vector<size_t> & overloaded_quad_pts_ref{this->overloaded_quad_pts};

    //! loop over all quad points and check if stress overcomes the threshold or
    //! not
    for (const auto && quad_pt_threshold : threshold_map.enumerate_indices()) {
      const auto & quad_pt{std::get<0>(quad_pt_threshold)};
      const Real & threshold{std::get<1>(quad_pt_threshold)};
      const auto & stress{stress_map[quad_pt]};
      // check if stress is larger than threshold,
      const Real sigma_eq{MatTB::compute_equivalent_von_Mises_stress(stress)};
      // if sigma_eq > threshold write quad_pt into Ccoord vector
      if ((sigma_eq > threshold) == 1) {
        // compute the global quad point id from the local quad_pt and the
        // local_quad_pt_id_offset
        overloaded_quad_pts_ref.push_back(quad_pt + local_quad_pt_id_offset);
      }
    }
    return overloaded_quad_pts_ref;
  }

  /* ---------------------------------------------------------------------- */
  //! Updates the eigen strain field of all overloaded pixels by doing a plastic
  //! increment into the deviatoric stress direction by an absolute value given
  //! by the plastic_increment_field
  template <Index_t DimM>
  decltype(auto)
      MaterialStochasticPlasticity<DimM>::update_eigen_strain_field(
          Cell & cell, Eigen::Ref<Vector_t> & stress_numpy_array) {
    muGrid::WrappedField<Real> stress_field{
        "temp input for stress field", cell.get_fields(), DimM * DimM,
        stress_numpy_array, QuadPtTag};
    return this->update_eigen_strain_field(stress_field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::update_eigen_strain_field(
      const muGrid::TypedFieldBase<Real> & stress_field) {
    muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>
        stress_map{stress_field};
    //! initialise strain
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    //! initialise plastic increment
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    //! loop over all overloaded_pixels
    for (const auto & pixel : this->overloaded_quad_pts) {
      //!  1.) compute plastic_strain_direction = σ_dev/Abs[σ_dev]
      const auto & stress_map_pixel{stress_map[pixel]};
      Eigen::Matrix<Real, DimM, DimM> stress_matrix_pixel = stress_map_pixel;
      const Eigen::Matrix<Real, DimM, DimM> deviatoric_stress{
          MatTB::compute_deviatoric(stress_matrix_pixel)};
      const Real equivalent_stress{
          MatTB::compute_equivalent_von_Mises_stress(stress_map_pixel)};
      const Eigen::Matrix<Real, DimM, DimM> plastic_strain_direction{
          deviatoric_stress / equivalent_stress};

      //!  2.) update eigen_strain_field
      eigen_strain_map[pixel] +=
          plastic_strain_direction * plastic_increment_map[pixel];
    }
    if (this->overloaded_quad_pts.size() > 0) {
      this->last_step_was_nonlinear = true;
    }
  }

  /* ---------------------------------------------------------------------- */
  //! archive_overloaded_pixels(), archives the overloaded pixels saved in
  //! this->overloaded_pixels to the input vector avalanche_history and empties
  //! overloaded_pixels.
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::archive_overloaded_quad_pts(
      std::list<std::vector<size_t>> & avalanche_history) {
    //!  1.) archive overloaded_pixels in avalanche_history
    avalanche_history.push_back(this->overloaded_quad_pts);
    //!  2.) clear overloaded pixels
    this->overloaded_quad_pts.clear();
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_STOCHASTIC_PLASTICITY_HH_
