/**
 * @file   projection_finite_strain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  implementation of standard finite strain projection operator
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
  ProjectionFiniteStrain(FFTEngine_ptr engine, Rcoord lengths)
    :Parent{std::move(engine), lengths, Formulation::finite_strain}
  {
    for (auto res: this->fft_engine->get_domain_resolutions()) {
      if (res % 2 == 0) {
      	throw ProjectionError
	  ("Only an odd number of gridpoints in each direction is supported");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrain<DimS, DimM>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);
    FFT_freqs<DimS> fft_freqs(this->fft_engine->get_domain_resolutions(),
                              this->domain_lengths);
    for (auto && tup: akantu::zip(*this->fft_engine, this->Ghat)) {
      const auto & ccoord = std::get<0> (tup);
      auto & G = std::get<1>(tup);
      auto xi = fft_freqs.get_unit_xi(ccoord);
      //! this is simplifiable using Curnier's Méthodes numériques, 6.69(c)
      G = Matrices::outer_under(Matrices::I2<DimM>(), xi*xi.transpose());
      // The commented block below corresponds to the original
      // definition of the operator in de Geus et
      // al. (https://doi.org/10.1016/j.cma.2016.12.032). However,
      // they use a bizarre definition of the double contraction
      // between fourth-order and second-order tensors that has a
      // built-in transpose operation (i.e., C = A:B <-> AᵢⱼₖₗBₗₖ =
      // Cᵢⱼ , note the inverted ₗₖ instead of ₖₗ), here, we define
      // the double contraction without the transposition. As a
      // result, the Projection operator produces the transpose of de
      // Geus's

      // for (Dim_t im = 0; im < DimS; ++im) {
      //   for (Dim_t j = 0; j < DimS; ++j) {
      //     for (Dim_t l = 0; l < DimS; ++l) {
      //       get(G, im, j, l, im) = xi(j)*xi(l);
      //     }
      //   }
      // }
    }
    if (this->get_subdomain_locations() == Ccoord{}) {
      this->Ghat[0].setZero();
    }
  }

  template class ProjectionFiniteStrain<twoD,   twoD>;
  template class ProjectionFiniteStrain<threeD, threeD>;
}  // muSpectre
