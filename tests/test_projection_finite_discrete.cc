/**
 * @file   test_projection_finite_discrete.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   16 Apr 2017
 *
 * @brief  tests for discrete finite strain projection operator
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

#include <random>

#include <libmufft/fft_utils.hh>
#include <libmufft/fft_engine_base.hh>

#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"

#include "test_projection.hh"

#include <Eigen/Dense>

using muFFT::DiscreteDerivative;
using muGrid::IterUnit;

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain_discrete);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
      ProjectionFixture<twoD, twoD, Squares<twoD>, DiscreteGradient<twoD>,
                        ProjectionFiniteStrain<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        DiscreteGradient<threeD>,
                        ProjectionFiniteStrain<threeD>>,
      ProjectionFixture<twoD, twoD, Squares<twoD>, DiscreteGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD>>,
      ProjectionFixture<
          twoD, twoD, Squares<twoD>, DiscreteGradient<twoD, TwoQuadPts>,
          ProjectionFiniteStrainFast<twoD, TwoQuadPts>, TwoQuadPts>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        DiscreteGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(
        fix::projector.initialise());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(discrete_derivative_1d_test) {
    constexpr double tol = 1e-6;

    // Upwind differences
    DiscreteDerivative stencil({2}, {0}, std::vector<Real>{-1, 1});
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 1).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 1).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0.5).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0.25).finished())
                .real() -
            -1.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(oneD) << 0.25).finished())
                .imag() -
            1.,
        tol);

    // Central differences
    DiscreteDerivative stencil2({3}, {-1}, std::vector<Real>{-0.5, 0, 0.5});
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 1).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 1).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0.5).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0.25).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(oneD) << 0.25).finished())
                .imag() -
            1.,
        tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(discrete_derivative_2d_test) {
    constexpr double tol = 1e-6;

    // Upwind differences
    DiscreteDerivative stencil({2, 1}, {0, 0}, std::vector<Real>{-1, 1});
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 1, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 1, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0.5, 0).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0.5, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0.5, 1).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier((DiscreteDerivative::Vector(twoD) << 0.5, 1).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier(
                   (DiscreteDerivative::Vector(twoD) << 0.25, 0).finished())
                .real() -
            -1.,
        tol);
    BOOST_CHECK_SMALL(
        stencil.fourier(
                   (DiscreteDerivative::Vector(twoD) << 0.25, 0).finished())
                .imag() -
            1.,
        tol);

    DiscreteDerivative stencil2{stencil.rollaxes()};
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(twoD) << 0, 1).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier((DiscreteDerivative::Vector(twoD) << 0, 1).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.5).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 1, 0.5).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 1, 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.25).finished())
                .real() -
            -1.,
        tol);
    BOOST_CHECK_SMALL(
        stencil2.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.25).finished())
                .imag() -
            1.,
        tol);

    // Averaged upwind differences
    DiscreteDerivative stencil3({2, 2}, {0, 0},
                                std::vector<Real>{-0.5, 0.5, -0.5, 0.5});
    DiscreteDerivative stencil4{stencil3.rollaxes()};
    BOOST_CHECK_SMALL(
        stencil4.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier((DiscreteDerivative::Vector(twoD) << 0, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier((DiscreteDerivative::Vector(twoD) << 1, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier((DiscreteDerivative::Vector(twoD) << 1, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.5).finished())
                .real() -
            -2.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0, 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 0.5).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 0.5).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 1).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.5, 1).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.25, 0).finished())
                .real() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 0.25, 0).finished())
                .imag() -
            0.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 1, 0.25).finished())
                .real() -
            -1.,
        tol);
    BOOST_CHECK_SMALL(
        stencil4.fourier(
                    (DiscreteDerivative::Vector(twoD) << 1, 0.25).finished())
                .imag() -
            1.,
        tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(even_grid_test) {
    using Engine = muFFT::FFTWEngine;
    using proj = ProjectionFiniteStrain<twoD>;
    auto nb_dof{2 * 2};
    auto engine = std::make_unique<Engine>(DynCcoord_t{2, 3});
    engine->create_plan(nb_dof);
    BOOST_CHECK_THROW(proj(std::move(engine), DynRcoord_t{4.3, 4.3}),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test, fix, fixlist,
                                   fix) {
    // create a first order central difference gradient field with a zero mean
    // gradient and verify that the projection preserves it.
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim},
        nb_quad{fix::nb_quad};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using FieldMap = muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim,
                                            mdim * nb_quad, IterUnit::Pixel>;
    using FieldMap1D =
        muGrid::MatrixFieldMap<Real, Mapping::Mut, 1, mdim, IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Gradient_t gradient{fix::GradientGiver::get_gradient()};

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, nb_quad);
    // displacement field
    muGrid::RealField & f_disp{
        fields.register_real_field("displacement", mdim, QuadPtTag)};
    // gradient of the displacement field
    muGrid::RealField & f_grad{
        fields.register_real_field("gradient", mdim * mdim, QuadPtTag)};
    // field for comparision
    muGrid::RealField & f_var{
        fields.register_real_field("working field", mdim * mdim, QuadPtTag)};

    FieldMap1D disp(f_disp);
    FieldMap grad(f_grad);
    FieldMap var(f_var);

    BOOST_TEST_CHECKPOINT("fields and maps constructed");

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    BOOST_TEST_CHECKPOINT("fields and maps initialised");

    muFFT::FFT_freqs<dim> freqs{fix::projector.get_nb_domain_grid_pts(),
                                fix::projector.get_domain_lengths()};

    Rcoord_t<mdim> delta_x{(fix::projector.get_domain_lengths() /
                            fix::projector.get_nb_domain_grid_pts())
                               .template get<mdim>()};

    // fill the displacement field with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto && d : disp) {
      Vector u;
      for (int k{0}; k < dim; ++k) {
        u[k] = dis(gen);
      }
      d.row(0) = u;
    }

    BOOST_TEST_CHECKPOINT("displacement field filled");

    // compute the gradient field in real space
    for (Dim_t quad{0}; quad < nb_quad; ++quad) {
      for (Dim_t i{0}; i < dim; ++i) {
        Dim_t k{quad * dim + i};
        for (Dim_t j{0}; j < dim; ++j) {
          auto derivative_op{
              std::dynamic_pointer_cast<muFFT::DiscreteDerivative>(
                  gradient[k])};
          // Storage order of gradient fields: We want to be able to iterate
          // over a gradient field using either QuadPts or Pixels iterators.
          // A quadrature point iterator returns a dim x dim matrix. A pixels
          // iterator must return a dim x dim * nb_quad matrix, since every-
          // thing is column major this matrix is just two dim x dim matrices
          // that are stored consecutive in memory. This means the components of
          // the displacement field, not the components of the gradient, must be
          // stored consecutive in memory and are the first index.
          derivative_op->apply(f_disp, j, f_grad, j + dim * k,
                               1.0 / delta_x[i]);
          derivative_op->apply(f_disp, j, f_var, j + dim * k, 1.0 / delta_x[i]);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("gradient field computed");

    fix::projector.initialise();
    fix::projector.apply_projection(f_var);

    BOOST_TEST_CHECKPOINT("projection applied");

    for (auto && tup : akantu::zip(
             fields.get_pixels().template get_dimensioned_pixels<mdim>(), grad,
             var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(ccoord, delta_x);
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
        std::cout << std::endl << "nb_grid_pts :" << std::endl;
        muGrid::operator<<(std::cout,
                           fix::projector.get_nb_subdomain_grid_pts())
            << std::endl;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(idempotent_test, fix, fixlist, fix) {
    // check if the discrete projection operator is still a projection operator.
    // Thus it has to be idempotent, G^2=G or G:G:test_field = G:test_field.
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim},
        nb_quad{fix::nb_quad};
    using Fields = muGrid::GlobalFieldCollection;
    using FieldMap = muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim,
                                            IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, nb_quad);
    muGrid::RealField & f_grad{
        fields.register_real_field("gradient", mdim * mdim, QuadPtTag)};
    muGrid::RealField & f_grad_test{
        fields.register_real_field("gradient_test", mdim * mdim, QuadPtTag)};
    FieldMap grad(f_grad);
    FieldMap grad_test(f_grad_test);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    for (auto && tup : akantu::zip(
             fields.get_pixels().template get_dimensioned_pixels<mdim>(), grad,
             grad_test)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & gt = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<mdim>());
      g.row(0) = vec.transpose() * cos(vec.dot(vec));
      gt.row(0) = g.row(0);
    }

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
      if (error >= tol) {
        std::cout
            << std::endl
            << "g - gt " << error << " , tol is " << tol << std::endl
            << "The discrete compatibility operator seems to be not idempotent!"
            << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
