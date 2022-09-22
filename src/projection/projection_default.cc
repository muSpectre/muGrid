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
  template <Index_t DimS, Index_t NbQuadPts>
  ProjectionDefault<DimS, NbQuadPts>::ProjectionDefault(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Gradient_t & gradient, const Weights_t & weights,
      const Formulation & form, const MeanControl & mean_control)
      : Parent{std::move(engine),
               lengths,
               static_cast<Index_t>(gradient.size()) / lengths.get_dim(),
               DimS * DimS,
               gradient,
               weights,
               form,
               mean_control},
        Gfield{this->fft_engine->get_fourier_field_collection()
                   .register_complex_field(
                       ProjectionDefault<DimS, NbQuadPts>::
                           prepare_field_unique_name(this->fft_engine,
                                                     "Projection Operator"),
                       DimS * DimS * NbQuadPts * DimS * DimS * NbQuadPts,
                       PixelTag)},
        Ghat{Gfield}, Ifield{this->fft_engine->get_fourier_field_collection()
                                 .register_complex_field(
                                     ProjectionDefault<DimS, NbQuadPts>::
                                         prepare_field_unique_name(
                                             this->fft_engine,
                                             "Integration Operator"),
                                     DimS * DimS * DimS * NbQuadPts, PixelTag)},
        Ihat{Ifield} {
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
  template <Index_t DimS, Index_t NbQuadPts>
  void ProjectionDefault<DimS, NbQuadPts>::apply_projection(Field_t & field) {
    if (!this->initialised) {
      throw ProjectionError("Applying a projection without having initialised "
                            "the projector is not supported.");
    }
    this->fft_engine->fft(field, this->work_space);
    Vector_map field_map{this->work_space};
    Real factor = this->fft_engine->normalisation();

    // weights
    Eigen::Matrix<muGrid::Real, DimS * DimS * NbQuadPts, 1> w;
    for (int q = 0; q < NbQuadPts; ++q) {
      for (int i = 0; i < DimS * DimS; ++i) {
        w(q * DimS * DimS + i) = this->weights[q];
      }
    }

    // projection (of the stress field!)
    for (auto && tup : akantu::zip(this->Ghat, field_map)) {
      auto && G{std::get<0>(tup)};
      auto && f{std::get<1>(tup)};
      f = factor * (G * f.cwiseProduct(w)).eval();
    }
    this->fft_engine->ifft(this->work_space, field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  typename ProjectionDefault<DimS, NbQuadPts>::Field_t &
  ProjectionDefault<DimS, NbQuadPts>::integrate(Field_t & grad) {
    //! iterable form of the integrator
    using Grad_map = muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS,
                                            DimS * NbQuadPts, IterUnit::Pixel>;
    //! vectorized version of the real-space positions
    using RealPotential_map =
        muGrid::T1FieldMap<Real, Mapping::Mut, DimS, IterUnit::Pixel>;

    // store average strain
    this->fft_engine->fft(grad, this->work_space);
    Grad_map grad_map{this->work_space};
    Real factor = this->fft_engine->normalisation();
    Eigen::Matrix<Real, DimS, DimS * NbQuadPts> avg_strain{grad_map[0].real() *
                                                           factor};
    // avg_strain(ξ=0) ← 0
    if (!(this->get_subdomain_locations() == Ccoord{})) {
      avg_strain.setZero();
    }
    avg_strain = this->get_communicator().sum(avg_strain);

    // This operation writes the integrated nonaffine displacement field into
    // the real space field "Node potential (in real space)" which then can be
    // fetched. So it is necessary that the real space potential field in
    // ProjectionDefault::integrate_nonaffine_displacements has the same name
    // as in ProjectionDefault::integrate.
    this->integrate_nonaffine_displacements(grad);
    auto & potential{this->fft_engine->fetch_or_register_real_space_field(
        "Node positions (in real space)", DimS)};

    // add average strain to positions
    RealPotential_map real_positions_map{potential};
    auto grid_spacing{this->domain_lengths / this->get_nb_domain_grid_pts()};
    for (auto && tup :
         akantu::zip(this->fft_engine->get_real_pixels(), real_positions_map)) {
      auto & c{std::get<0>(tup)};
      auto & p{std::get<1>(tup)};
      for (Index_t i{0}; i < DimS; ++i) {
        p += avg_strain.col(i) * c[i] * grid_spacing[i];
      }
    }
    return potential;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  typename ProjectionDefault<DimS, NbQuadPts>::Field_t &
  ProjectionDefault<DimS, NbQuadPts>::integrate_nonaffine_displacements(
      Field_t & grad) {
    //! vectorized version of the Fourier-space positions
    using FourierPotential_map =
        muGrid::T1FieldMap<Complex, Mapping::Mut, DimS, IterUnit::Pixel>;

    if (!this->initialised) {
      throw ProjectionError("Integrating a field without having initialised "
                            "the projector is not supported.");
    }

    // positions in Fourier space
    auto & potential_work_space{
        this->fft_engine->fetch_or_register_fourier_space_field(
            "Nodal nonaffine displacements (in Fourier space)", DimS)};
    this->fft_engine->fft(grad, this->work_space);
    Real factor = this->fft_engine->normalisation();

    // apply integrator
    Vector_map vector_map{this->work_space};
    FourierPotential_map fourier_positions_map{potential_work_space};
    for (auto && tup :
         akantu::zip(this->Ihat, vector_map, fourier_positions_map)) {
      auto & I{std::get<0>(tup)};
      auto & g{std::get<1>(tup)};
      auto & p{std::get<2>(tup)};
      p = factor * (I * g).eval();
    }
    auto & potential{this->fft_engine->fetch_or_register_real_space_field(
        "Node positions (in real space)", DimS)};
    this->fft_engine->ifft(potential_work_space, potential);
    return potential;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  Eigen::Map<MatrixXXc> ProjectionDefault<DimS, NbQuadPts>::get_operator() {
    return this->Gfield.eigen_pixel();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  std::array<Index_t, 2>
  ProjectionDefault<DimS, NbQuadPts>::get_strain_shape() const {
    return std::array<Index_t, 2>{DimS, DimS};
  }

  /* ---------------------------------------------------------------------- */
  template class ProjectionDefault<oneD, OneQuadPt>;
  template class ProjectionDefault<oneD, TwoQuadPts>;
  template class ProjectionDefault<oneD, FourQuadPts>;
  template class ProjectionDefault<oneD, FiveQuadPts>;
  template class ProjectionDefault<oneD, SixQuadPts>;
  template class ProjectionDefault<twoD, OneQuadPt>;
  template class ProjectionDefault<twoD, TwoQuadPts>;
  template class ProjectionDefault<twoD, FourQuadPts>;
  template class ProjectionDefault<twoD, FiveQuadPts>;
  template class ProjectionDefault<twoD, SixQuadPts>;
  template class ProjectionDefault<threeD, OneQuadPt>;
  template class ProjectionDefault<threeD, TwoQuadPts>;
  template class ProjectionDefault<threeD, FourQuadPts>;
  template class ProjectionDefault<threeD, FiveQuadPts>;
  template class ProjectionDefault<threeD, SixQuadPts>;
}  // namespace muSpectre
