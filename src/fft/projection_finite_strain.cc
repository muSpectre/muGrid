/**
 * file   projection_finite_strain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  implementation of standard finite strain projection operator
 *
 * @section LICENSE
 *
 * Copyright © 2017 Till Junge
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

#include "fft/projection_finite_strain.hh"
#include "fft/fftw_engine.hh"
#include "fft/fft_utils.hh"
#include "common/field_map.hh"
#include "common/tensor_algebra.hh"
#include "common/iterators.hh"

#include "Eigen/Dense"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionFiniteStrain<DimS, DimM>::
  ProjectionFiniteStrain(FFT_Engine_ptr engine)
    :Parent{std::move(engine), Formulation::finite_strain}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrain<DimS, DimM>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);
    FFT_freqs<DimS> fft_freqs(this->fft_engine->get_resolutions(),
                              this->fft_engine->get_lengths());
    for (auto && tup: akantu::zip(*this->fft_engine, this->Ghat)) {
      const auto & ccoord = std::get<0> (tup);
      auto & G = std::get<1>(tup);
      auto xi = fft_freqs.get_unit_xi(ccoord);
      //! this is simplifiable using Curnier's Méthodes numériques, 6.69(c)
      G = Matrices::outer_under(Matrices::I2<DimM>(), xi*xi.transpose());
      // for (Dim_t im = 0; im < DimS; ++im) {
      //   for (Dim_t j = 0; j < DimS; ++j) {
      //     for (Dim_t l = 0; l < DimS; ++l) {
      //       get(G, im, j, l, im) = xi(j)*xi(l);
      //     }
      //   }
      // }
    }
    this->Ghat[0].setZero();
  }

  template class ProjectionFiniteStrain<twoD,   twoD>;
  template class ProjectionFiniteStrain<threeD, threeD>;
}  // muSpectre
