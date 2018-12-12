/**
 * @file   material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   21 Feb 2018
 *
 * @brief  implementation for MaterialHyperElastoPlastic1
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "materials/material_hyper_elasto_plastic1.hh"

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  MaterialHyperElastoPlastic1<DimS, DimM>::MaterialHyperElastoPlastic1(
      std::string name, Real young, Real poisson, Real tau_y0, Real H)
      : Parent{name},
        plast_flow_field{
            make_statefield<StateField<ScalarField<LColl_t, Real>>>(
                "cumulated plastic flow εₚ", this->internal_fields)},
        F_prev_field{make_statefield<
            StateField<TensorField<LColl_t, Real, secondOrder, DimM>>>(
            "Previous placement gradient Fᵗ", this->internal_fields)},
        be_prev_field{make_statefield<
            StateField<TensorField<LColl_t, Real, secondOrder, DimM>>>(
            "Previous left Cauchy-Green deformation bₑᵗ",
            this->internal_fields)},
        young{young}, poisson{poisson}, lambda{Hooke::compute_lambda(young,
                                                                     poisson)},
        mu{Hooke::compute_mu(young, poisson)},
        K{Hooke::compute_K(young, poisson)}, tau_y0{tau_y0}, H{H},
        // the factor .5 comes from equation (18) in Geers 2003
        // (https://doi.org/10.1016/j.cma.2003.07.014)
        C{0.5 * Hooke::compute_C_T4(lambda, mu)},
        internal_variables{F_prev_field.get_map(), be_prev_field.get_map(),
                           plast_flow_field.get_map()} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic1<DimS, DimM>::save_history_variables() {
    this->plast_flow_field.cycle();
    this->F_prev_field.cycle();
    this->be_prev_field.cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic1<DimS, DimM>::initialise() {
    Parent::initialise();
    this->F_prev_field.get_map().current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->be_prev_field.get_map().current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->save_history_variables();
  }

  template class MaterialHyperElastoPlastic1<twoD, twoD>;
  template class MaterialHyperElastoPlastic1<twoD, threeD>;
  template class MaterialHyperElastoPlastic1<threeD, threeD>;
}  // namespace muSpectre
