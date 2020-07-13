/**
 * @file   projection_finite_strain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   05 Dec 2017
 *
 * @brief  implementation of the finite strain projection operator
 *
 * Copyright © 2017 Till Junge
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

#include "projection/projection_finite_strain.hh"

#include <libmugrid/iterators.hh>
#include <libmufft/fft_utils.hh>
#include <libmufft/fftw_engine.hh>

#include "Eigen/Dense"

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionFiniteStrain<DimS>::ProjectionFiniteStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      Gradient_t gradient)
      : Parent{std::move(engine), lengths, gradient,
               Formulation::finite_strain} {
    for (auto res : this->fft_engine->get_nb_domain_grid_pts()) {
      if (res % 2 == 0) {
        throw ProjectionError(
            "Only an odd number of grid points in each direction is supported");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionFiniteStrain<DimS>::ProjectionFiniteStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths)
      : ProjectionFiniteStrain{
            std::move(engine), lengths,
            muFFT::make_fourier_gradient(lengths.get_dim())} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  void
  ProjectionFiniteStrain<DimS>::initialise() {
    Parent::initialise();
    using FFTFreqs_t = muFFT::FFT_freqs<DimS>;
    using Vector_t = typename FFTFreqs_t::Vector;

    const auto & nb_domain_grid_pts =
        this->fft_engine->get_nb_domain_grid_pts();
    const Vector_t grid_spacing{eigen(
        (this->domain_lengths / nb_domain_grid_pts).template get<DimS>())};

    muFFT::FFT_freqs<DimS> fft_freqs(nb_domain_grid_pts);
    for (auto && tup : akantu::zip(this->fft_engine->get_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->Ghat)) {
      const auto & ccoord = std::get<0>(tup);
      auto & G = std::get<1>(tup);

      const Vector_t xi{(fft_freqs.get_xi(ccoord).array() /
                         eigen(nb_domain_grid_pts.template get<DimS>())
                             .array()
                             .template cast<Real>())
                            .matrix()};

      Eigen::Matrix<Complex, DimS, 1> diffop;
      for (int i = 0; i < DimS; ++i) {
        diffop(i) = this->gradient[i]->fourier(xi) / grid_spacing[i];
      }
      const Real norm2{diffop.squaredNorm()};

      const Eigen::Matrix<Complex, DimS, DimS> diff_mat{
          diffop * diffop.adjoint() / norm2};

      for (Dim_t im = 0; im < DimS; ++im) {
        for (Dim_t j = 0; j < DimS; ++j) {
          for (Dim_t l = 0; l < DimS; ++l) {
            get(G, im, j, im, l) = diff_mat(j, l);
          }
        }
      }
    }
    if (this->get_subdomain_locations() == Ccoord{}) {
      this->Ghat[0].setZero();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  std::unique_ptr<ProjectionBase> ProjectionFiniteStrain<DimS>::clone() const {
    return std::make_unique<ProjectionFiniteStrain>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->get_gradient());
  }

  template class ProjectionFiniteStrain<oneD>;
  template class ProjectionFiniteStrain<twoD>;
  template class ProjectionFiniteStrain<threeD>;
}  // namespace muSpectre
