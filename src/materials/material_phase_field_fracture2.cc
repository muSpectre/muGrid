/**
 * @file   material_phase_field_fracture2.cc
 *
 * @author W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>
 *
 * @date   02 Dec 2021
 *
 * @brief Material for solving the elasticity subproblem of a phase field
 *        fracture model using a decomposition of the elastic energy density
 *        into volumetric and deviatoric parts based on components of the
 *        (small-strain) strain tensor.  The phase field phi couples to the
 *        deviatoric part in all cases, and to the volumetric part when its
 *        sign is positive.  This decomposition was proposed in Amor et al.,
 *        2009.
 *
 * Copyright © 2021 W. Beck Andrews
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

#include "material_phase_field_fracture2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialPhaseFieldFracture2<DimM>::MaterialPhaseFieldFracture2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts,  const Real & ksmall)
      : Parent{name, spatial_dimension, nb_quad_pts},
        lambda_field{this->get_prefix() + "local first Lame constant",
                     *this->internal_fields, QuadPtTag},
        mu_field{this->get_prefix() +
                     "local second Lame constant (shear modulus)",
                 *this->internal_fields, QuadPtTag},
        phase_field(this->get_prefix() +
                     "local phase field",
                 *this->internal_fields, QuadPtTag), ksmall{ksmall} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture2<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture2<DimM>::add_pixel(const size_t & pixel,
                                               const Real & Young_modulus,
                                               const Real & Poisson_ratio,
                                               const Real & phase_field) {
    this->internal_fields->add_pixel(pixel);
    // store the first(lambda) and second(mu) Lame constant in the field
    Real lambda = Hooke::compute_lambda(Young_modulus, Poisson_ratio);
    Real mu = Hooke::compute_mu(Young_modulus, Poisson_ratio);
    this->lambda_field.get_field().push_back(lambda);
    this->mu_field.get_field().push_back(mu);
    this->phase_field.get_field().push_back(phase_field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture2<DimM>::set_youngs_modulus(
      const size_t & quad_pt_id, const Real & Youngs_modulus) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda_old = lambda_map[quad_pt_id];
    const Real & mu_old = mu_map[quad_pt_id];
    const Real Poisson_ratio = Hooke::compute_poisson(lambda_old, mu_old);

    // compute updated first and second lame constant (lambda and mu)
    const Real lambda_new =
        Hooke::compute_lambda(Youngs_modulus, Poisson_ratio);
    const Real mu_new = Hooke::compute_mu(Youngs_modulus, Poisson_ratio);

    // assign new values to fields
    lambda_map[quad_pt_id] = lambda_new;
    mu_map[quad_pt_id] = mu_new;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture2<DimM>::set_poisson_ratio(
      const size_t & quad_pt_id, const Real & Poisson_ratio) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute young from first and second lame constant (lambda and mu)
    const Real & lambda_old = lambda_map[quad_pt_id];
    const Real & mu_old = mu_map[quad_pt_id];
    const Real Youngs_modulus = Hooke::compute_young(lambda_old, mu_old);

    // compute updated first and second lame constant (lambda and mu)
    const Real lambda_new =
        Hooke::compute_lambda(Youngs_modulus, Poisson_ratio);
    const Real mu_new = Hooke::compute_mu(Youngs_modulus, Poisson_ratio);

    // assign new values to fields
    lambda_map[quad_pt_id] = lambda_new;
    mu_map[quad_pt_id] = mu_new;
  }

/* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture2<DimM>::set_phase_field(
      const size_t & quad_pt_id, const Real & phase_field) {
    auto && phase_field_map{this->phase_field.get_map()};

    phase_field_map[quad_pt_id] = phase_field;
  }


  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialPhaseFieldFracture2<DimM>::get_youngs_modulus(
      const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];
    return Hooke::compute_young(lambda, mu);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialPhaseFieldFracture2<DimM>::get_poisson_ratio(
      const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];
    return Hooke::compute_poisson(lambda, mu);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialPhaseFieldFracture2<DimM>::get_phase_field(
      const size_t & quad_pt_id) {
    auto && phase_field_map{this->phase_field.get_map()};
    return phase_field_map[quad_pt_id];
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialPhaseFieldFracture2<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & E, const Real & lambda,
      const Real & mu, const Real & phi, const Real & ksmall) -> T2_t {
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    auto && interp = (1.0-phi)*(1.0-phi)*(1.0-ksmall) + ksmall;
    Real trace = E.trace();
    Real K_prefactor = trace >= 0.0 ? interp : 1.0;
    T2_t stress{T2_t::Zero()};
    T2_t I{Matrices::I2<DimM>()};

    stress = K_prefactor*(lambda+2.0/3.0*mu)*trace*I + 2.0*mu*interp*
            (E - trace/3.0*I);

    return stress;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialPhaseFieldFracture2<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & E, const Real & lambda,
      const Real & mu, const Real & phi,  const Real & ksmall) ->
      std::tuple<T2_t, T4_t> {
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;
    T2_t stress{T2_t::Zero()};
    T4_t stress_tangent{T4_t::Zero()};

    if (phi == 0.0) {  // skip decomposition completely if phi==0.0
        stress_tangent = Hooke::compute_C_T4(lambda, mu);
        stress = Matrices::tensmult(stress_tangent, E);
        return std::make_tuple(stress, stress_tangent);
    }

    auto && interp = (1.0-phi)*(1.0-phi)*(1.0-ksmall) + ksmall;
    Real trace = E.trace();
    Real K_prefactor = trace >= 0.0 ? interp : 1.0;
    T2_t I{Matrices::I2<DimM>()};
    stress = K_prefactor*(lambda+2.0/3.0*mu)*trace*I +
             interp*2.0*mu*(E - trace/3.0*I);
    stress_tangent = Hooke::compute_C_T4(K_prefactor*(lambda+2.0/3.0*mu)-
             2.0/3.0*mu*interp, mu*interp);

    return std::make_tuple(stress, stress_tangent);
  }
  template class MaterialPhaseFieldFracture2<twoD>;
  template class MaterialPhaseFieldFracture2<threeD>;
}  // namespace muSpectre
