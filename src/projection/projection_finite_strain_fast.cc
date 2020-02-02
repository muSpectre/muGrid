/**
 * @file   projection_finite_strain_fast.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Dec 2017
 *
 * @brief  implementation for fast projection in finite strain
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

#include "projection/projection_finite_strain_fast.hh"

#include <libmufft/fft_utils.hh>
#include <libmugrid/iterators.hh>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  ProjectionFiniteStrainFast<DimS>::ProjectionFiniteStrainFast(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      Gradient_t gradient)
      : Parent{std::move(engine), lengths, Formulation::finite_strain},
        xi_field{this->projection_container.register_complex_field(
            "Projection Operator", DimS)},
        xis(xi_field), gradient{gradient} {
    for (auto res : this->fft_engine->get_nb_domain_grid_pts()) {
      if (res % 2 == 0) {
        throw ProjectionError(
            "Only an odd number of grid points in each direction is supported");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  ProjectionFiniteStrainFast<DimS>::ProjectionFiniteStrainFast(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths)
      : ProjectionFiniteStrainFast{
            std::move(engine), lengths,
            muFFT::make_fourier_gradient(lengths.get_dim())} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void ProjectionFiniteStrainFast<DimS>::initialise(
      const muFFT::FFT_PlanFlags & flags) {
    Parent::initialise(flags);

    using FFTFreqs_t = muFFT::FFT_freqs<DimS>;
    using Vector_t = typename FFTFreqs_t::Vector;

    const auto & nb_domain_grid_pts =
        this->fft_engine->get_nb_domain_grid_pts();

    const Vector_t grid_spacing{
        eigen(Rcoord(this->domain_lengths) / Ccoord(nb_domain_grid_pts))};

    FFTFreqs_t fft_freqs(nb_domain_grid_pts);
    for (auto && tup : akantu::zip(this->fft_engine->get_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->xis)) {
      const auto & ccoord = std::get<0>(tup);
      auto & xi = std::get<1>(tup);

      // compute phase (without the factor of 2 pi)
      const Vector_t phase{
          (fft_freqs.get_xi(ccoord).array() /
           eigen(Ccoord(nb_domain_grid_pts)).template cast<Real>().array())
              .matrix()};
      for (int i = 0; i < DimS; ++i) {
        xi[i] = this->gradient[i]->fourier(phase) / grid_spacing[i];
      }

      if (xi.norm() > 0) {
        xi /= xi.norm();
      }
    }

    if (this->fft_engine->is_active()) {
      // locations are also zero if the engine is not active
      if (this->get_subdomain_locations() == Ccoord{}) {
        this->xis[0].setZero();
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void ProjectionFiniteStrainFast<DimS>::apply_projection(Field_t & field) {
    Grad_map field_map{this->fft_engine->fft(field)};
    Real factor = this->fft_engine->normalisation();
    for (auto && tup : akantu::zip(this->xis, field_map)) {
      auto & xi{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * ((f * xi).eval() * xi.adjoint());
    }
    this->fft_engine->ifft(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  Eigen::Map<MatrixXXc> ProjectionFiniteStrainFast<DimS>::get_operator() {
    return this->xi_field.eigen_pixel();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  std::array<Dim_t, 2>
  ProjectionFiniteStrainFast<DimS>::get_strain_shape() const {
    return std::array<Dim_t, 2>{DimS, DimS * OneQuadPt};
  }

  template <Dim_t DimS>
  std::unique_ptr<ProjectionBase>
  ProjectionFiniteStrainFast<DimS>::clone() const {
    return std::make_unique<ProjectionFiniteStrainFast>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->get_gradient());
  }
  template class ProjectionFiniteStrainFast<oneD>;
  template class ProjectionFiniteStrainFast<twoD>;
  template class ProjectionFiniteStrainFast<threeD>;
}  // namespace muSpectre
