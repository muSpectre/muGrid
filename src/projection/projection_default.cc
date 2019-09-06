/**
 * @file   projection_default.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jan 2018
 *
 * @brief  Implementation default projection implementation
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

#include "projection/projection_default.hh"
#include <libmufft/fft_engine_base.hh>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  ProjectionDefault<DimS>::ProjectionDefault(
    FFTEngine_ptr engine, Rcoord lengths, Gradient_t gradient, Formulation form)
      : Parent{std::move(engine), lengths, gradient, form},
        Gfield{this->projection_container.register_complex_field(
          "Projection Operator", DimS * DimS * DimS * DimS)},
        Ghat{Gfield} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void ProjectionDefault<DimS>::apply_projection(Field_t & field) {
    Vector_map field_map{this->fft_engine->fft(field)};
    Real factor = this->fft_engine->normalisation();
    for (auto && tup : akantu::zip(this->Ghat, field_map)) {
      auto & G{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * (G * f).eval();
    }
    this->fft_engine->ifft(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  Eigen::Map<MatrixXXc> ProjectionDefault<DimS>::get_operator() {
    return this->Gfield.eigen_pixel();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  std::array<Dim_t, 2> ProjectionDefault<DimS>::get_strain_shape() const {
    return std::array<Dim_t, 2>{DimS, DimS};
  }

  /* ---------------------------------------------------------------------- */
  template class ProjectionDefault<twoD>;
  template class ProjectionDefault<threeD>;
}  // namespace muSpectre
