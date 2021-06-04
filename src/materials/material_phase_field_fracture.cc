/**
 * @file   material_phase_field_fracture.cc
 *
 * @author W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>
 *
 * @date   02 Feb 2021
 *
 * @brief Material for solving the elasticity subproblem of a phase field
 *        fracture model.  A phase field phi is coupled to the tensile part
 *        of the elastic energy of an isotropic material.  The decomposition
 *        of the (small strain) elastic energy into tensile and compressive
 *        strains is performed using the principal strains as proposed by
 *        Miehe et al. 2010.
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

#include "material_phase_field_fracture.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialPhaseFieldFracture<DimM>::MaterialPhaseFieldFracture(
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
  void MaterialPhaseFieldFracture<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialPhaseFieldFracture<DimM>::add_pixel(const size_t & pixel,
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
  void MaterialPhaseFieldFracture<DimM>::set_youngs_modulus(
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
  void MaterialPhaseFieldFracture<DimM>::set_poisson_ratio(
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
  void MaterialPhaseFieldFracture<DimM>::set_phase_field(
      const size_t & quad_pt_id, const Real & phase_field) {
    auto && phase_field_map{this->phase_field.get_map()};

    phase_field_map[quad_pt_id] = phase_field;
  }


  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialPhaseFieldFracture<DimM>::get_youngs_modulus(
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
  MaterialPhaseFieldFracture<DimM>::get_poisson_ratio(
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
  MaterialPhaseFieldFracture<DimM>::get_phase_field(
      const size_t & quad_pt_id) {
    auto && phase_field_map{this->phase_field.get_map()};
    return phase_field_map[quad_pt_id];
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialPhaseFieldFracture<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & E, const Real & lambda,
      const Real & mu, const Real & phi, const Real & ksmall) -> T2_t {
    using Vec_t = Eigen::Matrix<Real, DimM, 1>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    auto && interp = (1.0-phi)*(1.0-phi)*(1.0-ksmall) + ksmall;
    const muGrid::SelfAdjointDecomp_t<DimM> spectral_decomp{
        muGrid::spectral_decomposition(E)};
    const Vec_t & eig_vals{spectral_decomp.eigenvalues()};
    const T2_t & eig_vecs{spectral_decomp.eigenvectors()};
    Vec_t mu_prefactor{Vec_t::Zero()};
    T2_t stress{T2_t::Zero()};

    for (int j{0}; j < DimM; ++j) {
      mu_prefactor(j) = eig_vals(j) >= 0.0 ? interp : 1.0;
    }
    Real lambda_prefactor = eig_vals.sum() >= 0.0 ? interp : 1.0;
    for (int j{0}; j < DimM; ++j) {
      T2_t dyad = eig_vecs.col(j) * eig_vecs.col(j).transpose();
      stress += (lambda_prefactor*lambda*eig_vals.sum() +
          2*mu_prefactor(j)*mu*eig_vals(j)) * dyad;
    }
    if ((DimM == 2) and (eig_vals[0] * eig_vals[1] < 0.0)) {
      stress += 1e-16*mu*E;
    }
    if ((DimM == 3) and (not(
        (eig_vals[0] >= 0.0 and eig_vals[1] >= 0.0 and eig_vals[2] >= 0.0)
     or (eig_vals[0] < 0.0 and eig_vals[1] < 0.0 and eig_vals[2] < 0.0)))) {
      stress += 1e-16*mu*E;
    }
    return stress;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialPhaseFieldFracture<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & E, const Real & lambda,
      const Real & mu, const Real & phi,  const Real & ksmall) ->
      std::tuple<T2_t, T4_t> {
    using Vec_t = Eigen::Matrix<Real, DimM, 1>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;
    T2_t stress{T2_t::Zero()};
    T4_t stress_tangent{T4_t::Zero()};

    if (phi == 0.0) {  // skip decomposition completely if phi==0.0
        stress_tangent = Hooke::compute_C_T4(lambda, mu);
        stress = Matrices::tensmult(stress_tangent, E);
        return std::make_tuple(stress, stress_tangent);
    }

    const muGrid::SelfAdjointDecomp_t<DimM> spectral_decomp{
        muGrid::spectral_decomposition(E)};
    const Vec_t eig_vals{spectral_decomp.eigenvalues()};
    const T2_t & eig_vecs{spectral_decomp.eigenvectors()};
    auto && interp = (1.0-phi)*(1.0-phi)*(1.0-ksmall) + ksmall;
    Real trace = eig_vals.sum();
    Real lambda_prefactor = trace >= 0 ? interp : 1.0;
    Vec_t mu_prefactor{Vec_t::Zero()};
    for (int j{0}; j < DimM; ++j) {
      mu_prefactor(j) = eig_vals(j) >= 0 ? interp : 1.0;
    }
    switch (DimM) {
    case twoD: {
      // shortcuts for no decomposition
      if (eig_vals[0] * eig_vals[1] >= 0.0) {
        stress_tangent = Hooke::compute_C_T4(lambda_prefactor*lambda,
            lambda_prefactor*mu);
        stress = Matrices::tensmult(stress_tangent, E);
      } else {
        stress_tangent = Hooke::compute_C_T4(lambda_prefactor*lambda, 0.0);
        stress_tangent += 1e-16*mu*Matrices::Iiden<DimM>();
        stress += 1e-16*mu*E;
        for (int i{0}; i < DimM; ++i) {
          for (int j{0}; j < DimM; ++j) {
            T2_t dyad = eig_vecs.col(i) * eig_vecs.col(j).transpose();
            T4_t outerDyad = Matrices::outer(dyad, dyad.transpose());
            if (i == j) {
              stress_tangent += 2.0*mu_prefactor(j)* mu * outerDyad;
              stress += (lambda_prefactor*lambda*trace
                  + 2.0*mu_prefactor(i)*mu*eig_vals(i)) * dyad;
            } else {
              T4_t symmDyad = Matrices::outer(dyad+dyad.transpose(),
                  dyad+dyad.transpose());
              stress_tangent += mu_prefactor(i)*mu*eig_vals(i)
                  /(eig_vals(i) - eig_vals(j))*symmDyad;
            }
          }
        }
      }
      return std::make_tuple(stress, stress_tangent);
    }
    case threeD: {
      if ((eig_vals[0] >= 0.0 and eig_vals[1] >= 0.0 and eig_vals[2] >= 0.0)
          or (eig_vals[0] < 0.0 and eig_vals[1] < 0.0 and eig_vals[2] < 0.0)) {
        stress_tangent = Hooke::compute_C_T4(lambda_prefactor*lambda,
            lambda_prefactor*mu);
        stress = Matrices::tensmult(stress_tangent, E);
      } else {
        int diff_ind;
        if (eig_vals[2] >= 0.0 and eig_vals[1] >= 0) {
          diff_ind = 0;
        } else {
          diff_ind = 2;
        }
        stress_tangent = Hooke::compute_C_T4(lambda_prefactor*lambda, 0.0);
        stress_tangent += 1e-16*mu*Matrices::Iiden<DimM>();
        stress += 1e-16*mu*E;
        for (int i{0}; i < DimM; ++i) {
          for (int j{0}; j < DimM; ++j) {
            T2_t dyad = eig_vecs.col(i) * eig_vecs.col(j).transpose();
            if (i == j) {
              stress += (lambda_prefactor*lambda*trace
                  + 2.0*mu_prefactor(i)*mu*eig_vals(i)) * dyad;
              T4_t outerDyad = Matrices::outer(dyad, dyad.transpose());
              stress_tangent += 2.0*mu_prefactor(i)*mu * outerDyad;
            } else {
              T4_t symmDyad = Matrices::outer(dyad+dyad.transpose(),
                  dyad+dyad.transpose());
              if (i != diff_ind and j != diff_ind) {
                stress_tangent += mu_prefactor(i)*mu*symmDyad/2.0;
              } else {
                stress_tangent += mu_prefactor(i)*mu*eig_vals(i)
                    /(eig_vals(i) - eig_vals(j))*symmDyad;
              }
            }
          }
        }
      }
      return std::make_tuple(stress, stress_tangent);
    }
    default: {
      throw MaterialError("Only 2D and 3D supported");
      break;
    }
    }
  }
  template class MaterialPhaseFieldFracture<twoD>;
  template class MaterialPhaseFieldFracture<threeD>;
}  // namespace muSpectre
