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
  template <Index_t DimS, Index_t NbQuadPts>
  ProjectionSmallStrain<DimS, NbQuadPts>::ProjectionSmallStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Gradient_t & gradient, const Weights_t & weights,
      const MeanControl & mean_control)
      : Parent{std::move(engine), lengths, gradient, weights,
               Formulation::small_strain, mean_control} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  ProjectionSmallStrain<DimS, NbQuadPts>::ProjectionSmallStrain(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const MeanControl & mean_control)
      : ProjectionSmallStrain{std::move(engine), lengths,
                              muFFT::make_fourier_gradient(lengths.get_dim()),
                              {1},
                              mean_control} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t NbQuadPts>
  void ProjectionSmallStrain<DimS, NbQuadPts>::initialise() {
    Parent::initialise();
    using FFTFreqs_t = muFFT::FFT_freqs<DimS>;
    using Vector_t = typename FFTFreqs_t::Vector;

    const auto & nb_domain_grid_pts{this->fft_engine->get_nb_domain_grid_pts()};
    const Vector_t grid_spacing{eigen(
        (this->domain_lengths / nb_domain_grid_pts).template get<DimS>())};

    muFFT::FFT_freqs<DimS> fft_freqs(nb_domain_grid_pts);
    for (auto && tup : akantu::zip(this->fft_engine->get_fourier_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->Ghat, this->Ihat)) {
      const auto & ccoord = std::get<0>(tup);
      auto & G = std::get<1>(tup);
      auto & I = std::get<2>(tup);

      const Vector_t xi{(fft_freqs.get_xi(ccoord).array() /
                         eigen(nb_domain_grid_pts.template get<DimS>())
                             .array()
                             .template cast<Real>())
                            .matrix()};

      // compute derivative operator
      Eigen::Matrix<Complex, DimS * NbQuadPts, 1> diffop;
      for (Index_t quad = 0; quad < NbQuadPts; ++quad) {
        for (Index_t dim = 0; dim < DimS; ++dim) {
          Index_t i = quad * DimS + dim;
          diffop[i] = this->gradient[i]->fourier(xi) / grid_spacing[dim];
        }
      }
      const Real norm2{diffop.squaredNorm()};

      // matrices g and h
      const Eigen::Matrix<Complex, DimS * NbQuadPts, 1> int_vec{
          diffop.conjugate() / norm2};
      const Eigen::Matrix<Complex, DimS * NbQuadPts, DimS * NbQuadPts> proj_mat{
          diffop * int_vec.transpose()};

      Eigen::Matrix<Complex, DimS, DimS> sum_proj_mat;
      sum_proj_mat.setIdentity();
      for (Dim_t i{0}; i < DimS; ++i) {
        for (Dim_t j{0}; j < DimS; ++j) {
          for (Dim_t q{0}; q < NbQuadPts; ++q) {
            sum_proj_mat(i, j) += proj_mat(i + q * DimS, j + q * DimS);
          }
        }
      }
      const Eigen::Matrix<Complex, DimS, DimS> h_mat{sum_proj_mat.inverse()};

      // integration
      I.setZero();
      for (Dim_t theta{0}; theta < NbQuadPts; ++theta) {
        for (Dim_t i{0}; i < DimS; ++i) {
          for (Dim_t j{0}; j < DimS; ++j) {
            for (Dim_t k{0}; k < DimS; ++k) {
              I(i, j + (k + theta * DimS) * DimS) =
                  int_vec(j + theta * DimS) * h_mat(i, k) +
                  h_mat(i, j) * int_vec(k + theta * DimS);
            }
          }
        }
      }

      // projection
      G.setZero();
      for (Dim_t theta{0}; theta < NbQuadPts; theta++) {
        for (Dim_t lambda{0}; lambda < NbQuadPts; lambda++) {
          for (Dim_t i{0}; i < DimS; ++i) {
            for (Dim_t j{0}; j < DimS; ++j) {
              for (Dim_t l{0}; l < DimS; ++l) {
                for (Dim_t m{0}; m < DimS; ++m) {
                  G(i + (j + theta * DimS) * DimS,
                    m + (l + lambda * DimS) * DimS) =
                      0.5 * (proj_mat(i + theta * DimS, l + lambda * DimS) *
                                 h_mat(j, m) +
                             proj_mat(i + theta * DimS, m + lambda * DimS) *
                                 h_mat(j, l) +
                             proj_mat(j + theta * DimS, l + lambda * DimS) *
                                 h_mat(i, m) +
                             proj_mat(j + theta * DimS, m + lambda * DimS) *
                                 h_mat(i, l));
                }
              }
            }
          }
        }
      }
    }

    if (this->get_subdomain_locations() == Ccoord{}) {
      // Ghat (Project operator) is set to either 0ᵢⱼₖₗ or δᵢₖδⱼₗ (Ghat^*) based
      // on that either mean strain value or mean stress value is imposed on the
      // cell The formulation is Based on: An algorithm for stress and mixed
      // control in Galerkin-based FFT homogenization, by Lucarini, and Segurado
      // DOI: 10.1002/nme.6069
      switch (this->mean_control) {
      case MeanControl::StrainControl: {
        // Ghat(ξ=0) ← 0ᵢⱼₖₗ
        this->Ghat[0].setZero();
        break;
      }
      case MeanControl::StressControl: {
        // Ghat(ξ=0) ← δᵢₖδⱼₗ
        auto G{this->Ghat[0]};
        auto I_symm{Matrices::Isymm<DimS>()};
        for (Dim_t theta{0}; theta < NbQuadPts; theta++) {
          for (Dim_t lambda{0}; lambda < NbQuadPts; lambda++) {
            for (Dim_t i{0}; i < DimS; ++i) {
              for (Dim_t j{0}; j < DimS; ++j) {
                for (Dim_t l{0}; l < DimS; ++l) {
                  for (Dim_t m{0}; m < DimS; ++m) {
                    G(i + (j + theta * DimS) * DimS,
                      m + (l + lambda * DimS) * DimS) = get(I_symm, i, j, l, m);
                  }
                }
              }
            }
          }
        }
        break;
      }
      case MeanControl::MixedControl: {
        muGrid::RuntimeError("Mixed control projection is not implemented yet");
        break;
      }
      default: {
        throw muGrid::RuntimeError("Unknown value for mean_control value");
        break;
      }
      }
      // However, Ihat (integrator operator) is only set to 0
      // because it is not used in the solvers developed here so far, and
      // basically its only use case so far was to reconstruct the displacement
      // field from the strain field for visualization purposes.
      this->Ihat[0].setZero();
    }
  }

  template <Index_t DimS, Index_t NbQuadPts>
  std::unique_ptr<ProjectionBase>
  ProjectionSmallStrain<DimS, NbQuadPts>::clone() const {
    return std::make_unique<ProjectionSmallStrain>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->get_gradient(), this->get_weights());
  }

  template class ProjectionSmallStrain<oneD, OneQuadPt>;
  template class ProjectionSmallStrain<oneD, TwoQuadPts>;
  template class ProjectionSmallStrain<oneD, FourQuadPts>;
  template class ProjectionSmallStrain<oneD, FiveQuadPts>;
  template class ProjectionSmallStrain<oneD, SixQuadPts>;
  template class ProjectionSmallStrain<twoD, OneQuadPt>;
  template class ProjectionSmallStrain<twoD, TwoQuadPts>;
  template class ProjectionSmallStrain<twoD, FourQuadPts>;
  template class ProjectionSmallStrain<twoD, FiveQuadPts>;
  template class ProjectionSmallStrain<twoD, SixQuadPts>;
  template class ProjectionSmallStrain<threeD, OneQuadPt>;
  template class ProjectionSmallStrain<threeD, TwoQuadPts>;
  template class ProjectionSmallStrain<threeD, FourQuadPts>;
  template class ProjectionSmallStrain<threeD, FiveQuadPts>;
  template class ProjectionSmallStrain<threeD, SixQuadPts>;
}  // namespace muSpectre
