/**
 * @file   projection_approx_Green_operator.cc
 *
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   01 Feb 2020
 *
 * @brief  Discrete Green's function for constant material properties
 *
 * Copyright © 2020 Martin Ladecký
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

#include "projection/projection_approx_Green_operator.hh"

#include <libmufft/fft_utils.hh>

#include <iostream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionApproxGreenOperator<DimS>::ProjectionApproxGreenOperator(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
          C_ref_,
      Gradient_t gradient)
      : Parent{std::move(engine), lengths, gradient, Formulation::small_strain},
        C_ref_holder{std::make_unique<C_t>(C_ref_)}, C_ref{
                                                         *this->C_ref_holder} {
    for (auto res : this->fft_engine->get_nb_domain_grid_pts()) {
      if (res % 2 == 0) {
        throw ProjectionError(
            "Only an odd number of gridpoints in each direction is supported");
      }
    }
    if (C_ref_.rows() != DimS * DimS) {
      throw ProjectionError("Wrong size C_ref_");
    }
    if (C_ref_.cols() != DimS * DimS) {
      throw ProjectionError("Wrong size C_ref_");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  ProjectionApproxGreenOperator<DimS>::ProjectionApproxGreenOperator(
      muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
      const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
          C_ref)
      : ProjectionApproxGreenOperator{
            std::move(engine), lengths, C_ref,
            muFFT::make_fourier_gradient(lengths.get_dim())} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  void ProjectionApproxGreenOperator<DimS>::initialise() {
    Parent::initialise();
    ProjectionApproxGreenOperator<DimS>::reinitialise(this->C_ref);
  }
  /* ---------------------------------------------------------------------- */
  template <Index_t DimS>
  void ProjectionApproxGreenOperator<DimS>::reinitialise(
      const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
          C_ref_new) {
    this->C_ref = C_ref_new;
    using muGrid::get;
    muFFT::FFT_freqs<DimS> fft_freqs(
        Ccoord(this->fft_engine->get_nb_domain_grid_pts()),
        Rcoord(this->domain_lengths));
    for (auto && tup : akantu::zip(this->fft_engine->get_pixels()
                                       .template get_dimensioned_pixels<DimS>(),
                                   this->Ghat)) {
      const auto & ccoord{std::get<0>(tup)};  // pointer to
      auto & G{std::get<1>(tup)};             // pointer to
      auto xi{
          fft_freqs.get_xi(ccoord)};  // change: get non-normalised frequencies
                                      //   auto &pointer_to_C_new = C_ref_new;

      Eigen::Matrix<Real, DimS, DimS> A{
          Eigen::Matrix<Real, DimS, DimS>::Zero()};
      for (Dim_t i{0}; i < DimS; ++i) {
        for (Dim_t j{0}; j < DimS; ++j) {
          for (Dim_t l{0}; l < DimS; ++l) {
            for (Dim_t m{0}; m < DimS; ++m) {
              A(i, l) += get(this->C_ref, i, j, l, m) * xi(j) * xi(m);
            }
          }
        }
      }

      auto && N{A.inverse()}; /* */

      /* New operator begin*/

      for (Dim_t i{0}; i < DimS; ++i) {
        for (Dim_t j{0}; j < DimS; ++j) {
          for (Dim_t l{0}; l < DimS; ++l) {
            for (Dim_t m{0}; m < DimS; ++m) {
              Complex & g = get(G, i, j, l, m);

              g = 0.25 * (N(j, m) * xi(i) * xi(l) + N(j, l) * xi(i) * xi(m) +
                          N(i, m) * xi(j) * xi(l) + N(i, l) * xi(j) * xi(m));
            }
          }
        }
      }
    }
    /* New operator end*/

    if (this->get_subdomain_locations() == Ccoord{}) {
      this->Ghat[0].setZero();
    }
  }

  template <Index_t DimS>
  std::unique_ptr<ProjectionBase>
  ProjectionApproxGreenOperator<DimS>::clone() const {
    return std::make_unique<ProjectionApproxGreenOperator>(
        this->get_fft_engine().clone(), this->get_domain_lengths(),
        this->C_ref, this->get_gradient());
  }

  template class ProjectionApproxGreenOperator<oneD>;
  template class ProjectionApproxGreenOperator<twoD>;
  template class ProjectionApproxGreenOperator<threeD>;

}  // namespace muSpectre
