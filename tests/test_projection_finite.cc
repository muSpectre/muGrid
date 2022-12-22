/**
 * @file   test_projection_finite.cc
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

#include <Eigen/Dense>

#include <libmugrid/field_typed.hh>

#include <libmufft/fft_utils.hh>
#include <libmufft/pocketfft_engine.hh>

#include <projection/projection_finite_strain.hh>
#include <projection/projection_finite_strain_fast.hh>

#include "libmugrid/test_goodies.hh"

#include "test_projection.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain);

  /* ---------------------------------------------------------------------- */
  using fixlistRankTwo = boost::mpl::list<
      ProjectionFixture<twoD,
                        twoD,
                        Squares<twoD>,
                        FourierGradient<twoD>,
                        ProjectionFiniteStrain<twoD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Squares<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrain<threeD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<twoD,
                        twoD,
                        Sizes<twoD>,
                        FourierGradient<twoD>,
                        ProjectionFiniteStrain<twoD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Sizes<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrain<threeD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,

      ProjectionFixture<twoD,
                        twoD,
                        Squares<twoD>,
                        FourierGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Squares<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<twoD,
                        twoD,
                        Sizes<twoD>,
                        FourierGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Sizes<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>>;

  using fixlistRankOne = boost::mpl::list<
      ProjectionFixture<twoD,
                        twoD,
                        Squares<twoD>,
                        FourierGradient<twoD>,
                        ProjectionGradient<twoD, firstOrder>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Squares<threeD>,
                        FourierGradient<threeD>,
                        ProjectionGradient<threeD, firstOrder>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<twoD,
                        twoD,
                        Sizes<twoD>,
                        FourierGradient<twoD>,
                        ProjectionGradient<twoD, firstOrder>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>,
      ProjectionFixture<threeD,
                        threeD,
                        Sizes<threeD>,
                        FourierGradient<threeD>,
                        ProjectionGradient<threeD, firstOrder>,
                        OneQuadPt,
                        muFFT::PocketFFTEngine>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlistRankTwo, fix) {
    BOOST_CHECK_NO_THROW(fix::projector.initialise());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(even_grid_test) {
    using Engine = muFFT::PocketFFTEngine;
    using proj = ProjectionFiniteStrainFast<twoD>;
    auto nb_dof{2 * 2};
    auto engine = std::make_unique<Engine>(DynCcoord_t{2, 3});
    engine->create_plan(nb_dof);
    BOOST_CHECK_THROW(proj(std::move(engine), DynRcoord_t{4.3, 4.3}),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(full_gradient_preservation_test, fix,
                                   fixlistRankTwo, fix) {
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
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test_rank_two, fix,
                                   fixlistRankTwo, fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using FieldMap =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim, IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    muGrid::RealField & f_grad{
        fields.register_real_field("gradient", mdim * mdim, QuadPtTag)};
    muGrid::RealField & f_var{
        fields.register_real_field("working field", mdim * mdim, QuadPtTag)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fix::projector.initialise();

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    muFFT::FFT_freqs<dim> freqs{fix::projector.get_nb_domain_grid_pts(),
                                fix::projector.get_domain_lengths()};
    Vector k;
    for (Dim_t i = 0; i < dim; ++i) {
      // The wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i + 1) * 2 * muGrid::pi / fix::projector.get_domain_lengths()[i];
    }

    using muGrid::operator/;
    // start_field_iteration_snippet
    for (auto && tup :
         akantu::zip(fields.get_pixels().template get_dimensioned_pixels<dim>(),
                     grad, var)) {
      auto & ccoord = std::get<0>(tup);  // iterate from fields
      auto & g = std::get<1>(tup);       // iterate from grad
      auto & v = std::get<2>(tup);       // iterate from var

      // use iterate in arbitrary expressions
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<dim>());
      // do efficient linear algebra on iterates
      g.row(0) = k.transpose() *
                 cos(k.dot(vec));  // This is a plane wave with wave vector k in
                                   // real space. A valid gradient field.
      v.row(0) = g.row(0);
    }
    // end_field_iteration_snippet

    fix::projector.apply_projection(f_var);

    for (auto && tup :
         akantu::zip(fields.get_pixels().template get_dimensioned_pixels<dim>(),
                     grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<dim>());
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >= tol) {
        std::cout << std::endl << "grad_ref :" << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "ccoord :" << std::endl;
        muGrid::operator<<(std::cout, ccoord) << std::endl;
        std::cout << std::endl
                  << "vector :" << std::endl
                  << vec.transpose() << std::endl;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test_rank_one, fix,
                                   fixlistRankOne, fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using FieldMap =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, 1, IterUnit::SubPt>;
    using ScalarFieldMap_t =
        muGrid::ScalarFieldMap<Real, Mapping::Mut, IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    muGrid::RealField & f_grad{
        fields.register_real_field("gradient", mdim, QuadPtTag)};
    muGrid::RealField & f_primitive{
        fields.register_real_field("primitive", 1, QuadPtTag)};
    muGrid::RealField & f_var{
        fields.register_real_field("working field", mdim, QuadPtTag)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);
    ScalarFieldMap_t primitive(f_primitive);

    fix::projector.initialise();

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    muFFT::FFT_freqs<dim> freqs{fix::projector.get_nb_domain_grid_pts(),
                                fix::projector.get_domain_lengths()};
    Vector k;
    // arbitrary number to avoid zero values of sine wave at discretisation
    // points
    Real phase_shift{.25};
    for (Dim_t i = 0; i < dim; ++i) {
      // The wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i + 1) * 2 * muGrid::pi / fix::projector.get_domain_lengths()[i];
    }

    using muGrid::operator/;
    // start_field_iteration_snippet
    for (auto && tup :
         akantu::zip(fields.get_pixels().template get_dimensioned_pixels<dim>(),
                     grad, var, primitive)) {
      auto & ccoord = std::get<0>(tup);  // iterate from fields
      auto & g = std::get<1>(tup);       // iterate from grad
      auto & v = std::get<2>(tup);       // iterate from var
      auto & p = std::get<3>(tup);       // iterate from primitive

      // use iterate in arbitrary expressions
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<dim>());
      // do efficient linear algebra on iterates
      g = k * cos(k.dot(vec) +
                  phase_shift);  // This is a plane wave with wave vector k in
                                 // real space. A valid gradient field.
      p = sin(k.dot(vec) + phase_shift);
      v = g;
    }
    // end_field_iteration_snippet

    fix::projector.apply_projection(f_var);

    for (auto && tup :
         akantu::zip(fields.get_pixels().template get_dimensioned_pixels<dim>(),
                     grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<dim>());
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >= tol) {
        std::cout << std::endl << "grad_ref :" << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "ccoord :" << std::endl;
        muGrid::operator<<(std::cout, ccoord) << std::endl;
        std::cout << std::endl
                  << "vector :" << std::endl
                  << vec.transpose() << std::endl;
      }
    }

    BOOST_TEST_CHECKPOINT("Before nodal placement integration");
    auto && reconstructed_primitive{fix::projector.integrate(f_grad)};
    ScalarFieldMap_t r_primitive{reconstructed_primitive};

    for (auto && tup : akantu::zip(r_primitive, primitive)) {
      auto && reconstructed{std::get<0>(tup)};
      auto && original{std::get<1>(tup)};
      auto && error{muGrid::testGoodies::rel_error(reconstructed, original)};
      BOOST_CHECK_LT(error, tol);
      if (not(error < tol)) {
        std::cout << "reconstructed value: " << reconstructed
                  << " != " << original << ", the original value" << std::endl;
      }
    }

    BOOST_TEST_CHECKPOINT("Before nodal nonaffine displacement integration");
    auto && reconstructed_nonaff_primitive{
        fix::projector.integrate_nonaffine_displacements(f_grad)};
    ScalarFieldMap_t r_nonaff_primitive{reconstructed_nonaff_primitive};

    for (auto && tup : akantu::zip(r_nonaff_primitive, primitive)) {
      auto && nonaff_reconstructed{std::get<0>(tup)};
      auto && nonaff_original{std::get<1>(tup)};
      auto && error{muGrid::testGoodies::rel_error(nonaff_reconstructed,
                                                   nonaff_original)};
      BOOST_CHECK_LT(error, tol);
      if (not(error < tol)) {
        std::cout << "nonaffine reconstructed value: " << nonaff_reconstructed
                  << " != " << nonaff_original
                  << ", the nonaffine original value" << std::endl;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(idempotent_test, fix, fixlistRankTwo, fix) {
    // check if the exact projection operator is a valid projection operator.
    // Thus it has to be idempotent, G^2=G or G:G:test_field = G:test_field.
    // Note that this is the case up to the weights that are multiplied into
    // the projection.
    constexpr Dim_t sdim{fix::sdim}, mdim{fix::mdim};
    using Fields = muGrid::GlobalFieldCollection;
    using FieldT = muGrid::TypedField<Real>;
    using FieldMap =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim, IterUnit::SubPt>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    FieldT & f_grad{
        fields.register_real_field("gradient", mdim * mdim, QuadPtTag)};
    FieldT & f_grad_test{
        fields.register_real_field("gradient_test", mdim * mdim, QuadPtTag)};
    FieldMap grad(f_grad);
    FieldMap grad_test(f_grad_test);

    fields.initialise(fix::projector.get_nb_domain_grid_pts(),
                      fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    f_grad.eigen_vec().setRandom();
    f_grad_test.eigen_vec() = f_grad.eigen_vec();

    fix::projector.initialise();
    // apply projection once; G:f_grad
    fix::projector.apply_projection(f_grad);

    // apply projection twice; G:G:f_grad_test
    fix::projector.apply_projection(f_grad_test);
    fix::projector.apply_projection(f_grad_test);

    for (auto && tup : akantu::zip(grad, grad_test)) {
      auto & g = std::get<0>(tup);
      auto & gt = std::get<1>(tup);
      Real error = (g - gt).norm();
      BOOST_CHECK_LT(error, tol);
      if (not(error < tol)) {
        std::cout
            << std::endl
            << "The exact compatibility operator seems to be not idempotent!"
            << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
