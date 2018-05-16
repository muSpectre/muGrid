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
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "fft/projection_default.hh"
#include "fft/fft_engine_base.hh"


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionDefault<DimS, DimM>::ProjectionDefault(FFTEngine_ptr engine,
                                                   Rcoord lengths,
                                                   Formulation form)
    :Parent{std::move(engine), lengths, form},
     Gfield{make_field<Proj_t>("Projection Operator",
                               this->projection_container)},
     Ghat{Gfield}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionDefault<DimS, DimM>::apply_projection(Field_t & field) {
    Vector_map field_map{this->fft_engine->fft(field)};
    Real factor = this->fft_engine->normalisation();
    for (auto && tup: akantu::zip(this->Ghat, field_map)) {
      auto & G{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * (G*f).eval();
    }
    this->fft_engine->ifft(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  Eigen::Map<Eigen::ArrayXXd> ProjectionDefault<DimS, DimM>::get_operator() {
    return this->Gfield.dyn_eigen();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::array<Dim_t, 2> ProjectionDefault<DimS, DimM>::get_strain_shape() const {
    return std::array<Dim_t, 2>{DimM, DimM};
  }

  /* ---------------------------------------------------------------------- */
  template class ProjectionDefault<twoD,   twoD>;
  template class ProjectionDefault<threeD, threeD>;
}  // muSpectre
