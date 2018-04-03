/**
 * @file   projection_small_strain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jan 2018
 *
 * @brief  Implementation for ProjectionSmallStrain
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

#include "fft/projection_small_strain.hh"
#include "fft/fft_utils.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionSmallStrain<DimS, DimM>::
  ProjectionSmallStrain(FFTEngine_ptr engine, Rcoord lengths)
    : Parent{std::move(engine), lengths, Formulation::small_strain}
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
  void ProjectionSmallStrain<DimS, DimM>::initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);

    FFT_freqs<DimS> fft_freqs(this->fft_engine->get_domain_resolutions(),
                              this->domain_lengths);
    for (auto && tup: akantu::zip(*this->fft_engine, this->Ghat)) {
      const auto & ccoord = std::get<0> (tup);
      auto & G = std::get<1>(tup);
      auto xi = fft_freqs.get_unit_xi(ccoord);
      auto kron = [](const Dim_t i, const Dim_t j) -> Real{
        return (i==j) ? 1. : 0.;
      };
      for (Dim_t i{0}; i < DimS; ++i) {
        for (Dim_t j{0}; j < DimS; ++j) {
          for (Dim_t l{0}; l < DimS; ++l) {
            for (Dim_t m{0}; m < DimS; ++m ) {
              Real & g = get(G, i, j, l, m);
              g = 0.5* (xi(i) * kron(j, l) * xi(m) +
                        xi(i) * kron(j, m) * xi(l) +
                        xi(j) * kron(i, l) * xi(m) +
                        xi(j) * kron(i, m) * xi(l)) -
                xi(i)*xi(j)*xi(l)*xi(m);
            }
          }
        }
      }
    }
    if (this->get_subdomain_locations() == Ccoord{}) {
      this->Ghat[0].setZero();
    }
  }

  template class ProjectionSmallStrain<twoD,   twoD>;
  template class ProjectionSmallStrain<threeD, threeD>;
}  // muSpectre
