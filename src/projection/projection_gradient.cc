/**
 * @file   projection_gradient.cc
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

#include "projection/projection_gradient.hh"

#include <libmufft/fft_utils.hh>
#include <libmugrid/iterators.hh>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::ProjectionGradient(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Gradient_t & gradient, const Weights_t & weights,
      const MeanControl & mean_control)
      : Parent{std::move(engine),
               lengths,
               static_cast<Index_t>(gradient.size()) / lengths.get_dim(),
               muGrid::ipow(DimS, GradientRank),
               gradient,
               weights,
               Formulation::finite_strain,
               mean_control},
        proj_field{"Projection Operator",
                   this->fft_engine->get_fourier_field_collection(), PixelTag},
        int_field{"Integration Operator",
                  this->fft_engine->get_fourier_field_collection(), PixelTag} {
    if (this->get_dim() != DimS) {
      std::stringstream message{};
      message << "Dimension mismatch: this projection is templated with "
                 "the spatial dimension "
              << DimS << ", but the FFT engine has the spatial dimension "
              << this->get_dim() << ".";
      throw ProjectionError{message.str()};
    }
    if (this->nb_quad_pts != NbQuadPts) {
      std::stringstream error;
      error << "Deduced number of quadrature points (= " << this->nb_quad_pts
            << ") differs from template argument (= " << NbQuadPts << ").";
      throw ProjectionError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::ProjectionGradient(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const MeanControl & mean_control)
      : ProjectionGradient{std::move(engine), lengths,
                           muFFT::make_fourier_gradient(lengths.get_dim()),
                           {1},
                           mean_control} {
    if (NbQuadPts != OneQuadPt) {
      throw ProjectionError(
          "Default constructor uses Fourier gradient "
          "which can only be used with a singe quadrature point");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  void ProjectionGradient<DimS, GradientRank, NbQuadPts>::initialise() {
    Parent::initialise();

    using FFTFreqs_t = muFFT::FFT_freqs<DimS>;
    using Vector_t = typename FFTFreqs_t::Vector;

    const auto & nb_domain_grid_pts =
        this->fft_engine->get_nb_domain_grid_pts();

    const Vector_t grid_spacing{
        eigen(Rcoord(this->domain_lengths) / Ccoord(nb_domain_grid_pts))};

    FFTFreqs_t fft_freqs(nb_domain_grid_pts);
    for (auto && tup :
         akantu::zip(this->fft_engine->get_fourier_pixels()
                         .template get_dimensioned_pixels<DimS>(),
                     this->proj_field.get_map(), this->int_field.get_map())) {
      const auto & ccoord = std::get<0>(tup);
      auto & proj_op = std::get<1>(tup);
      auto & int_op = std::get<2>(tup);

      // compute phase (without the factor of 2 pi)
      const Vector_t phase{
          (fft_freqs.get_xi(ccoord).array() /
           eigen(Ccoord(nb_domain_grid_pts)).template cast<Real>().array())
              .matrix()};
      for (Index_t quad = 0; quad < NbQuadPts; ++quad) {
        for (Index_t dim = 0; dim < DimS; ++dim) {
          Index_t i = quad * DimS + dim;
          // ξ(i)
          proj_op[i] = this->gradient[i]->fourier(phase) / grid_spacing[dim];
        }
      }

      int_op = proj_op.conjugate();
      auto n{proj_op.squaredNorm()};
      if (n > 0) {
        proj_op /= sqrt(n);
        int_op /= n;
      }
    }

    if (this->fft_engine->has_grid_pts() and
        (this->get_subdomain_locations() == Ccoord{})) {
      this->proj_field.get_map()[0].setZero();
      this->int_field.get_map()[0].setZero();

      switch (this->mean_control) {
      case MeanControl::StrainControl: {
        this->zero_freq_proj_holder->setZero();
        break;
      }
      case MeanControl::StressControl: {
        this->zero_freq_proj_holder->setIdentity();
        break;
      }
      case MeanControl::MixedControl: {
        muGrid::RuntimeError("Mixed control projection is not implemented yet");
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown value for mean_control value");
        break;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  void ProjectionGradient<DimS, GradientRank, NbQuadPts>::apply_projection(
      Field_t & field) {
    if (!this->initialised) {
      throw ProjectionError("Applying a projection without having initialised"
                            "the projector is not supported.");
    }
    // Storage order of gradient fields: We want to be able to iterate over a
    // gradient field using either QuadPts or Pixels iterators. A quadrature
    // point iterator returns a dim x dim matrix. A pixels iterator must

    // a dim x dim * nb_quad matrix, since every-thing is column major this
    // matrix is just two dim x dim matrices that are stored consecutive in
    // memory. This means the components of the displacement field, not the
    // gradient direction, must be stored consecutive in memory and are the
    // first index.
    this->fft_engine->fft(field, this->work_space);
    Grad_map field_map{this->work_space};
    Real factor{this->fft_engine->normalisation()};
    PixelVec_t zero_freq_projected;
    if (this->get_subdomain_locations() == Ccoord{}) {
      zero_freq_projected =
          factor * (this->zero_freq_proj * PixelVec_map(field_map[0].data()));
    }

    // weights
    Eigen::Matrix<muGrid::Real, NbGradRow, DimS * NbQuadPts> w;
    for (int j = 0; j < NbGradRow; ++j) {
      for (int q = 0; q < NbQuadPts; ++q) {
        for (int i = 0; i < DimS; ++i) {
          w(j, q * DimS + i) = this->weights[q];
        }
      }
    }

    // projection (of the stress field!)
    for (auto && tup : akantu::zip(this->proj_field.get_map(), field_map)) {
      auto & proj_op{std::get<0>(tup)};
      auto & f{std::get<1>(tup)};
      f = factor * ((f.cwiseProduct(w) *
                     proj_op.conjugate()).eval() * proj_op.transpose());
    }
    if (this->get_subdomain_locations() == Ccoord{}) {
      PixelVec_map(field_map[0].data()) = zero_freq_projected;
    }
    this->fft_engine->ifft(this->work_space, field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  auto
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::integrate(Field_t & grad)
      -> Field_t & {
    //! vectorized version of the real-space potential
    using RealPotential_map =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, NbPrimitiveRow,
                               NbPrimitiveCol, IterUnit::SubPt>;

    // store average strain
    this->fft_engine->fft(grad, this->work_space);
    Grad_map grad_map{this->work_space};
    Real factor = this->fft_engine->normalisation();
    Eigen::Matrix<Real, NbGradRow, NbGradCol> avg_grad{grad_map[0].real() *
                                                       factor};
    if (!(this->get_subdomain_locations() == Ccoord{})) {
      avg_grad.setZero();
    }
    avg_grad = this->get_communicator().sum(avg_grad);

    // This operation writes the integrated nonaffine displacement field into
    // the real space field "Node potential (in real space)" which then can be
    // fetched. So it is necessary that the real space potential field in
    // ProjectionGradient::integrate_nonaffine_displacements has the same name
    // as in ProjectionGradient::integrate.
    this->integrate_nonaffine_displacements(grad);
    auto & potential{this->fft_engine->fetch_or_register_real_space_field(
        "Node potential (in real space)", NbPrimitiveRow * NbPrimitiveCol)};

    // add average strain to potential
    RealPotential_map real_potential_map{potential};
    auto grid_spacing{this->domain_lengths / this->get_nb_domain_grid_pts()};
    for (auto && tup :
         akantu::zip(this->fft_engine->get_real_pixels(), real_potential_map)) {
      auto & c{std::get<0>(tup)};
      auto & p{std::get<1>(tup)};
      for (Index_t i{0}; i < DimS; ++i) {
        p += avg_grad.col(i) * c[i] * grid_spacing[i];
      }
    }
    return potential;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  auto ProjectionGradient<DimS, GradientRank, NbQuadPts>::
      integrate_nonaffine_displacements(Field_t & grad) -> Field_t & {
    //! vectorized version of the Fourier-space potential
    using FourierPotential_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, NbPrimitiveRow,
                               NbPrimitiveCol, IterUnit::SubPt>;

    if (!this->initialised) {
      throw ProjectionError("Integrating a field without having initialised "
                            "the projector is not supported.");
    }
    // potential in Fourier space
    auto & potential_work_space{
        this->fft_engine->fetch_or_register_fourier_space_field(
            "Node potential (in Fourier space)",
            NbPrimitiveRow * NbPrimitiveCol)};
    this->fft_engine->fft(grad, this->work_space);
    Grad_map grad_map{this->work_space};
    FourierPotential_map fourier_potential_map{potential_work_space};
    Real factor = this->fft_engine->normalisation();

    // apply integrator
    for (auto && tup : akantu::zip(this->int_field.get_map(), grad_map,
                                   fourier_potential_map)) {
      auto & int_op{std::get<0>(tup)};
      auto & g{std::get<1>(tup)};
      auto & p{std::get<2>(tup)};
      p = factor * (g * int_op).eval();
    }
    auto & potential{this->fft_engine->fetch_or_register_real_space_field(
        "Node potential (in real space)", NbPrimitiveRow * NbPrimitiveCol)};
    this->fft_engine->ifft(potential_work_space, potential);
    return potential;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  Eigen::Map<MatrixXXc>
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::get_operator() {
    return this->proj_field.get_field().eigen_pixel();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  std::array<Index_t, 2>
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::get_strain_shape() const {
    return std::array<Index_t, 2>{DimS, DimS};
  }

  template <Index_t DimS, Index_t GradientRank, Index_t NbQuadPts>
  std::unique_ptr<ProjectionBase>
  ProjectionGradient<DimS, GradientRank, NbQuadPts>::clone() const {
    return std::make_unique<ProjectionGradient>(this->get_fft_engine().clone(),
                                                this->get_domain_lengths(),
                                                this->get_gradient(),
                                                this->get_weights());
  }

  template class ProjectionGradient<oneD, firstOrder, OneQuadPt>;
  template class ProjectionGradient<oneD, firstOrder, TwoQuadPts>;
  template class ProjectionGradient<oneD, firstOrder, FourQuadPts>;
  template class ProjectionGradient<oneD, firstOrder, FiveQuadPts>;
  template class ProjectionGradient<oneD, firstOrder, SixQuadPts>;
  template class ProjectionGradient<twoD, firstOrder, OneQuadPt>;
  template class ProjectionGradient<twoD, firstOrder, TwoQuadPts>;
  template class ProjectionGradient<twoD, firstOrder, FourQuadPts>;
  template class ProjectionGradient<twoD, firstOrder, FiveQuadPts>;
  template class ProjectionGradient<twoD, firstOrder, SixQuadPts>;
  template class ProjectionGradient<threeD, firstOrder, OneQuadPt>;
  template class ProjectionGradient<threeD, firstOrder, TwoQuadPts>;
  template class ProjectionGradient<threeD, firstOrder, FourQuadPts>;
  template class ProjectionGradient<threeD, firstOrder, FiveQuadPts>;
  template class ProjectionGradient<threeD, firstOrder, SixQuadPts>;

  template class ProjectionGradient<oneD, secondOrder, OneQuadPt>;
  template class ProjectionGradient<oneD, secondOrder, TwoQuadPts>;
  template class ProjectionGradient<oneD, secondOrder, FourQuadPts>;
  template class ProjectionGradient<oneD, secondOrder, FiveQuadPts>;
  template class ProjectionGradient<oneD, secondOrder, SixQuadPts>;
  template class ProjectionGradient<twoD, secondOrder, OneQuadPt>;
  template class ProjectionGradient<twoD, secondOrder, TwoQuadPts>;
  template class ProjectionGradient<twoD, secondOrder, FourQuadPts>;
  template class ProjectionGradient<twoD, secondOrder, FiveQuadPts>;
  template class ProjectionGradient<twoD, secondOrder, SixQuadPts>;
  template class ProjectionGradient<threeD, secondOrder, OneQuadPt>;
  template class ProjectionGradient<threeD, secondOrder, TwoQuadPts>;
  template class ProjectionGradient<threeD, secondOrder, FourQuadPts>;
  template class ProjectionGradient<threeD, secondOrder, FiveQuadPts>;
  template class ProjectionGradient<threeD, secondOrder, SixQuadPts>;
}  // namespace muSpectre
