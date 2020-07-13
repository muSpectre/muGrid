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
  template <Index_t DimS, Index_t NbQuadPts>
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::ProjectionFiniteStrainFast(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Gradient_t & gradient)
      : Parent{std::move(engine), lengths,
               static_cast<Index_t>(gradient.size())/lengths.get_dim(),
               DimS*DimS,
               Formulation::finite_strain},
        xi_field{"Projection Operator",
                 this->fft_engine->get_fourier_field_collection(), PixelTag},
        gradient{gradient} {
    if (this->nb_quad_pts != NbQuadPts) {
      std::stringstream error;
      error << "Deduced number of quadrature points (= " << this->nb_quad_pts
            << ") differs from template argument (= " << NbQuadPts << ").";
      throw ProjectionError(error.str());
    }
    for (auto res : this->fft_engine->get_nb_domain_grid_pts()) {
      if (res % 2 == 0) {
        throw ProjectionError(
            "Only an odd number of grid points in each direction is supported");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::ProjectionFiniteStrainFast(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths)
      : ProjectionFiniteStrainFast{
          std::move(engine), lengths,
          muFFT::make_fourier_gradient(lengths.get_dim())} {
            if (NbQuadPts != OneQuadPt) {
              throw ProjectionError("Default constructor uses Fourier gradient "
                  "which can only be used with a singe quadrature point");
            }
          }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  void ProjectionFiniteStrainFast<DimS, NbQuadPts>::initialise() {
    Parent::initialise();

    using FFTFreqs_t = muFFT::FFT_freqs<DimS>;
    using Vector_t = typename FFTFreqs_t::Vector;

    const auto & nb_domain_grid_pts =
        this->fft_engine->get_nb_domain_grid_pts();

    const Vector_t grid_spacing{
        eigen(Rcoord(this->domain_lengths) / Ccoord(nb_domain_grid_pts))};

    FFTFreqs_t fft_freqs(nb_domain_grid_pts);
    for (auto && tup : akantu::zip(this->fft_engine->get_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->xi_field.get_map())) {
      const auto & ccoord = std::get<0>(tup);
      auto & xi = std::get<1>(tup);

      // compute phase (without the factor of 2 pi)
      const Vector_t phase{
          (fft_freqs.get_xi(ccoord).array() /
           eigen(Ccoord(nb_domain_grid_pts)).template cast<Real>().array())
              .matrix()};
      for (Index_t quad = 0; quad < NbQuadPts; ++quad) {
        for (Index_t dim = 0; dim < DimS; ++dim) {
          Index_t i = quad * DimS + dim;
          xi[i] = this->gradient[i]->fourier(phase) / grid_spacing[dim];
        }
      }

      if (xi.norm() > 0) {
        xi /= xi.norm();
      }
    }

    if (this->fft_engine->is_active()) {
      // locations are also zero if the engine is not active
      if (this->get_subdomain_locations() == Ccoord{}) {
        this->xi_field.get_map()[0].setZero();
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  void ProjectionFiniteStrainFast<DimS, NbQuadPts>::apply_projection(
      Field_t & field) {
    if (!this->initialised) {
      throw ProjectionError("Applying a projection without having initialised "
                            "the projector is not supported.");
    }
    // Storage order of gradient fields: We want to be able to iterate over a
    // gradient field using either QuadPts or Pixels iterators. A quadrature
    // point iterator returns a dim x dim matrix. A pixels iterator must return
    // a dim x dim * nb_quad matrix, since every-thing is column major this
    // matrix is just two dim x dim matrices that are stored consecutive in
    // memory. This means the components of the displacement field, not the
    // gradient direction, must be stored consecutive in memory and are the
    // first index.
    this->fft_engine->fft(field, this->work_space);
    Grad_map field_map{this->work_space};
    Real factor = this->fft_engine->normalisation();
    for (auto && tup : akantu::zip(this->xi_field.get_map(), field_map)) {
      auto & xi{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * ((f * xi.conjugate()).eval() * xi.transpose());
    }
    this->fft_engine->ifft(this->work_space, field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  Eigen::Map<MatrixXXc>
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::get_operator() {
    return this->xi_field.get_field().eigen_pixel();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  const muFFT::Gradient_t &
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::get_gradient() const {
    return this->gradient;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  std::array<Index_t, 2>
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::get_strain_shape() const {
    return std::array<Index_t, 2>{DimS, DimS * OneQuadPt};
  }

  template <Index_t DimS, Index_t NbQuadPts>
  std::unique_ptr<ProjectionBase>
  ProjectionFiniteStrainFast<DimS, NbQuadPts>::clone() const {
    return std::make_unique<ProjectionFiniteStrainFast>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->get_gradient());
  }

  template class ProjectionFiniteStrainFast<oneD, OneQuadPt>;
  template class ProjectionFiniteStrainFast<oneD, TwoQuadPts>;
  template class ProjectionFiniteStrainFast<oneD, FourQuadPts>;
  template class ProjectionFiniteStrainFast<twoD, OneQuadPt>;
  template class ProjectionFiniteStrainFast<twoD, TwoQuadPts>;
  template class ProjectionFiniteStrainFast<twoD, FourQuadPts>;
  template class ProjectionFiniteStrainFast<threeD, OneQuadPt>;
  template class ProjectionFiniteStrainFast<threeD, TwoQuadPts>;
  template class ProjectionFiniteStrainFast<threeD, FourQuadPts>;
}  // namespace muSpectre
