/**
 * @file   mpi_test_projection_finite.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Dec 2017
 *
 * @brief  tests for standard finite strain projection operator
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 50

#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "mpi_test_projection.hh"

#include <libmufft/fft_utils.hh>
#include <libmufft/fftw_engine.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

#include <Eigen/Dense>

namespace muFFT {
  using muSpectre::ProjectionFiniteStrain;
  using muSpectre::ProjectionFiniteStrainFast;
  using muSpectre::tol;

  BOOST_AUTO_TEST_SUITE(mpi_projection_finite_strain);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
#ifdef WITH_FFTWMPI
      ProjectionFixture<twoD, twoD, Squares<twoD>, ProjectionFiniteStrain<twoD>,
                        FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionFiniteStrain<threeD>, FFTWMPIEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>, ProjectionFiniteStrain<twoD>,
                        FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionFiniteStrain<threeD>, FFTWMPIEngine<threeD>>,

      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        ProjectionFiniteStrainFast<twoD>, FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionFiniteStrainFast<threeD>,
                        FFTWMPIEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>,
                        ProjectionFiniteStrainFast<twoD>, FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionFiniteStrainFast<threeD>,
                        FFTWMPIEngine<threeD>>,
#endif
#ifdef WITH_PFFT
      ProjectionFixture<twoD, twoD, Squares<twoD>, ProjectionFiniteStrain<twoD>,
                        PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionFiniteStrain<threeD>, PFFTEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>, ProjectionFiniteStrain<twoD>,
                        PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionFiniteStrain<threeD>, PFFTEngine<threeD>>,

      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        ProjectionFiniteStrainFast<twoD>, PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionFiniteStrainFast<threeD>, PFFTEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>,
                        ProjectionFiniteStrainFast<twoD>, PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionFiniteStrainFast<threeD>, PFFTEngine<threeD>>,
#endif
      ProjectionFixture<twoD, twoD, Squares<twoD>, ProjectionFiniteStrain<twoD>,
                        FFTWEngine<twoD>, false>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    if (fix::is_parallel || fix::projector.get_communicator().size() == 1) {
      BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
    }
    std::cout << "Hello world!" << std::endl;
  }

  BOOST_FIXTURE_TEST_CASE(full_gradient_preservation_test, fix, fixlist, fix) {
    if (!fix::is_parallel || fix::projector.get_communicator().size() > 1) {
      return;
    }
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection<sdim>;
    using RealFieldT = muGrid::RealField;
    using FourierFieldT = muGrid::ComplexField;
    using FieldMap = muGrid::MatrixFieldMap<Real, false, mdim, mdim>;
    using FieldMapGradFourier =
        muGrid::MatrixFieldMap<Complex, false, mdim, mdim>;
    using FieldMapDispFourier = muGrid::MatrixFieldMap<Complex, false, mdim, 1>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{1};
    RealFieldT & f_disp{
        fields.template register_field<RealFieldT>("displacement", mdim)};
    RealFieldT & f_grad{
        fields.template register_field<RealFieldT>("gradient", mdim * mdim)};
    RealFieldT & f_var{fields.template register_field<RealFieldT>(
        "working field", mdim * mdim)};

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    // create a random displacement field with zero mean
    f_disp.eigen_vec().setRandom();
    disp -= disp.mean();

    // differentiate displacement field in fourier space
    auto && fft_engine{fix::projector.get_fft_engine()};
    FourierFieldT & f_disp_q{
        fft_engine.fetch_or_register_fourier_space_field("fourier disp", mdim)};
    FourierFieldT & f_grad_q{fft_engine.fetch_or_register_fourier_space_field(
        "fourier grad", mdim * mdim)};
    FieldMapGradFourier grad_q{f_grad_q};
    FieldMapDispFourier disp_q{f_disp_q};
    fix::projector.initialise(FFT_PlanFlags::estimate);

    fft_engine->fft(f_disp, f_disp_q);

    BOOST_CHECK_LE(tol, f_disp_q.eigen_vec().squared_norm());

    auto && nb_domain_grid_pts{fix::projector.get_nb_domain_grid_pts()};
    muFFT::FFT_freqs<dim> fft_freqs{nb_domain_grid_pts};
    const Vector_t grid_spacing{
        eigen((fft_engine->get_domain_lengths() / nb_domain_grid_pts)
                  .template get<dim>())};
    auto && gradient_operator{fft_engine.get_gradient()};
    for (auto && ccoord_disp_grad :
         akantu::zip(fft_engine->get_fourier_pixels()
                         .template get_dimensioned_pixels<DimS>(),
                     disp_q, grad_q)) {
      auto && ccoord{std::get<0>(ccoord_disp_grad)};
      auto && u{std::get<1>(ccoord_disp_grad)};
      auto && g{std::get<2>(ccoord_disp_grad)};
      const Vector_t xi{(fft_freqs.get_xi(ccoord).array() /
                         eigen(nb_domain_grid_pts.template get<DimS>())
                             .array()
                             .template cast<Real>())
                            .matrix()};
      // compute derivative operator
      Eigen::Matrix<Complex, dim, 1> diffop;
      for (Index_t dim = 0; dim < DimS; ++dim) {
        Index_t i = quad * DimS + dim;
        diffop(i) = gradient_operator[i]->fourier(xi) / grid_spacing[dim];
      }
      g = diffop.dot(u);
    }

    fft_engine->ifft(f_grad_q, f_grad);

    f_var = f_grad;

    fix::projector.apply_projection(f_var);

    Real error { testGoodies::rel_error(fvar.eigen_vec(), f_grad.eigen_vec()) }
    if (not(error < = tol)) {
      std::cout << "Projection failure:" << std::endl
                << "reference gradient = " << std::endl
                << f_grad.eigen_vec().transpose() << std::endl
                << "projected gradient = " << std::endl
                << f_var.eigen_vec().transpose() << std::endl;
    }
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test, fix, fixlist,
                                   fix) {
    if (!fix::is_parallel || fix::projector.get_communicator().size() > 1) {
      return;
    }
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection<sdim>;
    using RealFieldT = muGrid::RealField;
    using FieldMap = muGrid::MatrixFieldMap<Real, false, mdim, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{1};
    RealFieldT & f_grad{
        fields.template register_field<RealFieldT>("gradient", mdim * mdim)};
    RealFieldT & f_var{fields.template register_field<RealFieldT>(
        "working field", mdim * mdim)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());
    grad.initialise();
    var.initialise();

    Vector k;
    for (Dim_t i = 0; i < dim; ++i) {
      // the wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i + 1) * 2 * muGrid::pi / fix::projector.get_domain_lengths()[i];
    }

    using muGrid::operator/;
    for (auto && tup : akantu::zip(fields.get_pixels(), grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, fix::projector.get_domain_lengths() /
                      fix::projector.get_nb_domain_grid_pts());
      g.row(0) = k.transpose() * cos(k.dot(vec));
      v.row(0) = g.row(0);
    }

    fix::projector.initialise(FFT_PlanFlags::estimate);
    fix::projector.apply_projection(f_var);

    using muGrid::operator<<;
    for (auto && tup : akantu::zip(fields.get_pixels(), grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >= tol) {
        Vector vec = muGrid::CcoordOps::get_vector(
            ccoord, fix::projector.get_domain_lengths() /
                        fix::projector.get_nb_domain_grid_pts());
        std::cout << std::endl << "grad_ref :" << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl
                  << "ccoord :" << std::endl
                  << ccoord << std::endl;
        std::cout << std::endl
                  << "vector :" << std::endl
                  << vec.transpose() << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muFFT
