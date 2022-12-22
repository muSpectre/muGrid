/**
 * @file   mpi_test_projection_small.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   16 Jan 2018
 *
 * @brief  tests for standard small strain projection operator
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

#include "projection/projection_small_strain.hh"
#include "test_projection.hh"
#include "libmugrid/test_goodies.hh"

#include <libmufft/fft_utils.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

#include <Eigen/Dense>

namespace muSpectre {
  using muSpectre::ProjectionSmallStrain;
  using muSpectre::tol;
  using muGrid::operator/;
  using muGrid::operator<<;

  BOOST_AUTO_TEST_SUITE(mpi_projection_small_strain);

  using fixlist = boost::mpl::list<
#ifdef WITH_FFTWMPI
      MPIProjectionFixture<twoD, twoD, Squares<twoD>, FourierGradient<twoD>,
                           ProjectionSmallStrain<twoD>, 1,
                           muFFT::FFTWMPIEngine>,
      MPIProjectionFixture<
          threeD, threeD, Squares<threeD>, FourierGradient<threeD>,
          ProjectionSmallStrain<threeD>, 1, muFFT::FFTWMPIEngine>,
      MPIProjectionFixture<twoD, twoD, Sizes<twoD>, FourierGradient<twoD>,
                           ProjectionSmallStrain<twoD>, 1,
                           muFFT::FFTWMPIEngine>,
      MPIProjectionFixture<
          threeD, threeD, Sizes<threeD>, FourierGradient<threeD>,
          ProjectionSmallStrain<threeD>, 1, muFFT::FFTWMPIEngine>
#endif
#if defined(WITH_FFTWMPI) && defined(WITH_PFFT)
      ,
#endif /* defined(WITH_FFTWMPI) && defined(WITH_PFFT)*/
#ifdef WITH_PFFT
      MPIProjectionFixture<twoD, twoD, Squares<twoD>, FourierGradient<twoD>,
                           ProjectionSmallStrain<twoD>, 1, muFFT::PFFTEngine>,
      MPIProjectionFixture<threeD, threeD, Squares<threeD>,
                           FourierGradient<threeD>,
                           ProjectionSmallStrain<threeD>, 1, muFFT::PFFTEngine>,
      MPIProjectionFixture<twoD, twoD, Sizes<twoD>, FourierGradient<twoD>,
                           ProjectionSmallStrain<twoD>, 1, muFFT::PFFTEngine>,
      MPIProjectionFixture<threeD, threeD, Sizes<threeD>,
                           FourierGradient<threeD>,
                           ProjectionSmallStrain<threeD>, 1, muFFT::PFFTEngine>
#endif
      >;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(full_gradient_preservation_test, fix,
                                   fixlist, fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using RealFieldT = muGrid::RealField;
    using FourierFieldT = muGrid::TypedFieldBase<Complex>;
    using FieldMap =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, 1, IterUnit::Pixel>;
    using FieldMapGradFourier =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, mdim, mdim,
                               IterUnit::Pixel>;
    using FieldMapDispFourier =
        muGrid::MatrixFieldMap<Complex, Mapping::Const, mdim, 1,
                               IterUnit::Pixel>;
    Fields fields{1};
    RealFieldT & f_disp{fields.register_real_field("displacement", mdim)};
    RealFieldT & f_grad{fields.register_real_field("gradient", mdim * mdim)};
    RealFieldT & f_var{
        fields.register_real_field("working field", mdim * mdim)};

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    // create a random displacement field with zero mean
    f_disp.eigen_vec().setRandom();

    FieldMap disp{f_disp};
    disp -= disp.mean();

    // differentiate displacement field in fourier space
    auto && fft_engine{fix::projector.get_fft_engine()};
    FourierFieldT & f_disp_q{
        fft_engine.fetch_or_register_fourier_space_field("fourier disp", mdim)};
    FourierFieldT & f_grad_q{fft_engine.fetch_or_register_fourier_space_field(
        "fourier grad", mdim * mdim)};
    FieldMapGradFourier grad_q{f_grad_q};
    FieldMapDispFourier disp_q{f_disp_q};
    fix::projector.initialise();

    fft_engine.fft(f_disp, f_disp_q);

    BOOST_CHECK_LE(tol, f_disp_q.eigen_vec().squaredNorm());

    auto && nb_domain_grid_pts{fix::projector.get_nb_domain_grid_pts()};
    muFFT::FFT_freqs<dim> fft_freqs{nb_domain_grid_pts};
    using Vector_t = typename muFFT::FFT_freqs<dim>::Vector;
    const Vector_t grid_spacing(
        eigen((fix::projector.get_domain_lengths() / nb_domain_grid_pts)
                  .template get<dim>()));
    auto && gradient_operator{fix::projector.get_gradient()};
    for (auto && ccoord_disp_grad :
         akantu::zip(fft_engine.get_fourier_pixels()
                         .template get_dimensioned_pixels<dim>(),
                     disp_q, grad_q)) {
      auto && ccoord{std::get<0>(ccoord_disp_grad)};
      auto && u{std::get<1>(ccoord_disp_grad)};
      auto && g{std::get<2>(ccoord_disp_grad)};
      const Vector_t xi((fft_freqs.get_xi(ccoord).array() /
                         eigen(nb_domain_grid_pts.template get<dim>())
                             .array()
                             .template cast<Real>())
                            .matrix());

      // compute derivative operator
      Eigen::Matrix<Complex, dim, 1> diffop;
      for (Index_t i = 0; i < dim; ++i) {
        diffop[i] = gradient_operator[i]->fourier(xi) / grid_spacing[i];
      }
      g = (diffop * u.transpose()).transpose();
      // We need to add I to the term, because this field has a net
      // zero gradient, which leads to a net -I strain
      g = 0.5 * (g.transpose() + g).eval();
    }

    fft_engine.ifft(f_grad_q, f_grad);

    f_var = f_grad;

    fix::projector.apply_projection(f_var);

    Real error{
        muGrid::testGoodies::rel_error(f_var.eigen_vec(), f_grad.eigen_vec())};
    if (not(error <= tol)) {
      std::cout << "Projection failure:" << std::endl
                << "reference gradient = " << std::endl
                << f_grad.eigen_vec().transpose() << std::endl
                << "projected gradient = " << std::endl
                << f_var.eigen_vec().transpose() << std::endl;
    }
    BOOST_CHECK_LE(error, tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(gradient_preservation_test, fix, fixlist,
                                   fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using FieldT = muGrid::RealField;
    using FieldMap =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim, IterUnit::Pixel>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{1};
    FieldT & f_grad{fields.register_real_field("strain", mdim * mdim)};
    FieldT & f_var{fields.register_real_field("working field", mdim * mdim)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    Vector k;
    for (Dim_t i = 0; i < dim; ++i) {
      // the wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i + 1) * 2 * muGrid::pi / fix::projector.get_domain_lengths()[i];
    }

    for (auto && tup : akantu::zip(fields.get_pixels(), grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord.template get<mdim>(), (fix::projector.get_domain_lengths() /
                                        fix::projector.get_nb_domain_grid_pts())
                                           .template get<mdim>());
      g.row(0) << k.transpose() * cos(k.dot(vec));

      // We need to add I to the term, because this field has a net
      // zero gradient, which leads to a net -I strain
      g = 0.5 * (g.transpose() + g).eval();
      v = g;
    }

    fix::projector.initialise();
    fix::projector.apply_projection(f_var);

    constexpr muGrid::Verbosity verbose{muGrid::Verbosity::Silent};
    for (auto && tup : akantu::zip(fields.get_pixels(), grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord.template get<dim>(), (fix::projector.get_domain_lengths() /
                                       fix::projector.get_nb_domain_grid_pts())
                                          .template get<dim>());
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if ((error >= tol) || verbose > muGrid::Verbosity::Silent) {
        std::cout << std::endl << "grad_ref :" << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl
                  << "ccoord :" << std::endl
                  << ccoord << std::endl;
        std::cout << std::endl
                  << "vector :" << std::endl
                  << vec.transpose() << std::endl;
        std::cout << "means:" << std::endl
                  << "<strain>:" << std::endl
                  << grad.mean() << std::endl
                  << "<proj>:" << std::endl
                  << var.mean();
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
