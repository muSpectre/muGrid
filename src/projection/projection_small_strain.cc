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

#include "projection/projection_small_strain.hh"
#include <libmufft/fft_utils.hh>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionSmallStrain<DimS>::ProjectionSmallStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      Gradient_t gradient)
      : Parent{std::move(engine), lengths, gradient,
               Formulation::small_strain} {
    for (auto res : this->fft_engine->get_nb_domain_grid_pts()) {
      if (res % 2 == 0) {
        throw ProjectionError(
            "Only an odd number of gridpoints in each direction is supported");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionSmallStrain<DimS>::ProjectionSmallStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths)
      : ProjectionSmallStrain{std::move(engine), lengths,
                              muFFT::make_fourier_gradient(lengths.get_dim())} {
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  void
  ProjectionSmallStrain<DimS>::initialise() {
    using muGrid::get;
    Parent::initialise();

    muFFT::FFT_freqs<DimS> fft_freqs(
        Ccoord(this->fft_engine->get_nb_domain_grid_pts()),
        Rcoord(this->domain_lengths));
    for (auto && tup : akantu::zip(this->fft_engine->get_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->Ghat)) {
      const auto & ccoord = std::get<0>(tup);

      auto & G = std::get<1>(tup);
      auto xi = fft_freqs.get_unit_xi(ccoord);
      auto kron = [](const Dim_t i, const Dim_t j) -> Real {
        return (i == j) ? 1. : 0.;
      };
      for (Dim_t i{0}; i < DimS; ++i) {
        for (Dim_t j{0}; j < DimS; ++j) {
          for (Dim_t l{0}; l < DimS; ++l) {
            for (Dim_t m{0}; m < DimS; ++m) {
              Complex & g = get(G, i, j, l, m);
              g = 0.5 *
                      (xi(i) * kron(j, l) * xi(m) + xi(i) * kron(j, m) * xi(l) +
                       xi(j) * kron(i, l) * xi(m) +
                       xi(j) * kron(i, m) * xi(l)) -
                  xi(i) * xi(j) * xi(l) * xi(m);
            }
          }
        }
      }
    }
    if (this->get_subdomain_locations() == Ccoord{}) {
      this->Ghat[0].setZero();
    }
  }

  template <Index_t DimS>
  std::unique_ptr<ProjectionBase> ProjectionSmallStrain<DimS>::clone() const {
    return std::make_unique<ProjectionSmallStrain>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->get_gradient());
  }
  template class ProjectionSmallStrain<oneD>;
  template class ProjectionSmallStrain<twoD>;
  template class ProjectionSmallStrain<threeD>;
}  // namespace muSpectre
