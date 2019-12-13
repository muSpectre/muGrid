/**
 * @file   material_linear_elastic4.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de
 *
 * @date   15 March 2018
 *
 * @brief linear elastic material with distribution of stiffness properties.
 *        In difference to material_linear_elastic3 two Lame constants are
 *        stored per pixel instead of the whole elastic matrix C.
 *        Uses the MaterialMuSpectre facilities to keep it simple.
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

#include "material_linear_elastic4.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  MaterialLinearElastic4<DimM>::MaterialLinearElastic4(
      const std::string & name, const Dim_t & spatial_dimension,
      const Dim_t & nb_quad_pts)
      : Parent{name, spatial_dimension, nb_quad_pts},
        lambda_field{"local first Lame constant", this->internal_fields},
        mu_field("local second Lame constant(shear modulus)",
                 this->internal_fields) {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void MaterialLinearElastic4<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw std::runtime_error(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void MaterialLinearElastic4<DimM>::add_pixel(const size_t & pixel,
                                               const Real & Young_modulus,
                                               const Real & Poisson_ratio) {
    this->internal_fields.add_pixel(pixel);
    // store the first(lambda) and second(mu) Lame constant in the field
    Real lambda = Hooke::compute_lambda(Young_modulus, Poisson_ratio);
    Real mu = Hooke::compute_mu(Young_modulus, Poisson_ratio);
    this->lambda_field.get_field().push_back(lambda);
    this->mu_field.get_field().push_back(mu);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void MaterialLinearElastic4<DimM>::initialise() {
    Parent::initialise();
// TODO: comment
    // this->lambda_field.get_map().initialise();
    // this->mu_field.get_map().initialise();
  }

  template class MaterialLinearElastic4<twoD>;
  template class MaterialLinearElastic4<threeD>;

}  // namespace muSpectre
