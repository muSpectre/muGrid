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
#include <libmufft/fft_utils.hh>
#include <libmufft/fft_engine_base.hh>

#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"

#include "test_projection.hh"

#include <Eigen/Dense>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain_discrete);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        DiscreteGradient<twoD>,
                        ProjectionFiniteStrain<twoD, twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        DiscreteGradient<threeD>,
                        ProjectionFiniteStrain<threeD, threeD>>,

      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        DiscreteGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD, twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        DiscreteGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD, threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(
        fix::projector.initialise(muFFT::FFT_PlanFlags::estimate));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(discrete_derivative_1d_test) {
    constexpr double tol = 1e-6;

    // Upwind differences
    DiscreteDerivative<oneD> stencil({2}, {0}, std::vector<Real>{-1, 1});
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 1)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 1)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.5)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.5)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.25)
        .finished()).real() - -1., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.25)
        .finished()).imag() - 1., tol);

    // Central differences
    DiscreteDerivative<oneD> stencil2({3}, {-1},
                                      std::vector<Real>{-0.5, 0, 0.5});
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 1)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 1)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.5)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.5)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.25)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<oneD>::Vector() << 0.25)
        .finished()).imag() - 1., tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(discrete_derivative_2d_test) {
    constexpr double tol = 1e-6;

    // Upwind differences
    DiscreteDerivative<twoD> stencil({2, 1}, {0, 0}, std::vector<Real>{-1, 1});
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 1)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 1)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.25, 0)
        .finished()).real() - -1., tol);
    BOOST_CHECK_SMALL(stencil.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.25, 0)
        .finished()).imag() - 1., tol);

    DiscreteDerivative<twoD> stencil2{stencil.rollaxes()};
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 1)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 1)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0.5)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0.5)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0.5)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0.5)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0.25)
        .finished()).real() - -1., tol);
    BOOST_CHECK_SMALL(stencil2.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0.25)
        .finished()).imag() - 1., tol);

    // Averaged upwind differences
    DiscreteDerivative<twoD> stencil3({2, 2}, {0, 0},
                                      std::vector<Real>{-0.5, -0.5, 0.5, 0.5});
    DiscreteDerivative<twoD> stencil4{stencil3.rollaxes()};
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 1, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0.5)
        .finished()).real() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 0.5)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 1)
        .finished()).real() - -2., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.5, 1)
        .finished()).imag() - 0., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.25, 0)
        .finished()).real() - -1., tol);
    BOOST_CHECK_SMALL(stencil4.fourier(
        (DiscreteDerivative<twoD>::Vector() << 0.25, 0)
        .finished()).imag() - 1., tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(even_grid_test) {
    using Engine = muFFT::FFTWEngine<twoD>;
    using proj = ProjectionFiniteStrain<twoD, twoD>;
    auto engine = std::make_unique<Engine>(Ccoord_t<twoD>{2, 3}, 2 * 2);
    BOOST_CHECK_THROW(proj(std::move(engine), Rcoord_t<twoD>{4.3, 4.3}),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test, fix, fixlist,
                                   fix) {
    // create a first order central difference gradient field with a zero mean
    // gradient and verify that the projection preserves it.
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection<sdim>;
    using FieldT = muGrid::TensorField<Fields, Real, secondOrder, mdim>;
    using FieldT1D = muGrid::TensorField<Fields, Real, firstOrder, mdim>;
    using FieldMap = muGrid::MatrixFieldMap<Fields, Real, mdim, mdim>;
    using FieldMap1D = muGrid::MatrixFieldMap<Fields, Real, oneD, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{};
    // displacement field
    FieldT1D & f_disp{muGrid::make_field<FieldT1D>("displacement", fields)};
    // gradient of the displacement field
    FieldT & f_grad{muGrid::make_field<FieldT>("gradient", fields)};
    // field for comparision
    FieldT & f_var{muGrid::make_field<FieldT>("working field", fields)};

    FieldMap1D disp(f_disp);
    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());
    muFFT::FFT_freqs<dim> freqs{fix::projector.get_nb_domain_grid_pts(),
                                fix::projector.get_domain_lengths()};

    auto delta_x{muGrid::operator/(fix::projector.get_domain_lengths(),
                                   fix::projector.get_nb_domain_grid_pts())};
    // fill the displacement field
    for (auto && tup : akantu::zip(fields, disp)) {
      auto & ccoord = std::get<0>(tup);
      auto & d = std::get<1>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(ccoord, delta_x);
      Vector devided;
      for (int k = 0; k < dim; k++) {
        devided[k] = vec[k] / fix::projector.get_nb_domain_grid_pts()[k];
      }
      d.row(0) =
          1.0 / (2.0 * muGrid::pi) * (2.0 * muGrid::pi * devided.array()).sin();
    }

    // compute the gradient field
    for (auto && tup : akantu::zip(fields, grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);

      // compute first order central difference derivative
      Ccoord_t<sdim> ccoord_min;
      ccoord_min.fill(0);
      Ccoord_t<sdim> ccoord_max;
      ccoord_max = fix::projector.get_nb_domain_grid_pts();
      for (Dim_t i = 0; i < dim; i++) {
        for (Dim_t j = 0; j < dim; j++) {
          auto ccoord_minus_delta = ccoord;
          auto ccoord_plus_delta = ccoord;
          ccoord_minus_delta[j] -= 1;
          ccoord_plus_delta[j] += 1;
          // fix for periodic boundary conditions
          if (ccoord_minus_delta[j] < ccoord_min[j]) {
            ccoord_minus_delta[j] = ccoord_max[j]-1;
          }
          if (ccoord_plus_delta[j] >= ccoord_max[j]) {
            ccoord_plus_delta[j] = ccoord_min[j];
          }
          g.row(j) = (disp[ccoord_plus_delta] - disp[ccoord_minus_delta]) /
                     (2 * delta_x[j]);
        }
      }
      v = g;
    }

    fix::projector.initialise(muFFT::FFT_PlanFlags::estimate);
    fix::projector.apply_projection(f_var);

    for (auto && tup : akantu::zip(fields, grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, delta_x);
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
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(idempotent_test, fix, fixlist,
                                   fix) {
    // check if the discrete projection operator is still a projection operator.
    // Thus it has to be idempotent, G^2=G or G:G:test_field = G:test_field.
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    using Fields = muGrid::GlobalFieldCollection<sdim>;
    using FieldT = muGrid::TensorField<Fields, Real, secondOrder, mdim>;
    using FieldMap = muGrid::MatrixFieldMap<Fields, Real, mdim, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{};
    FieldT & f_grad{muGrid::make_field<FieldT>("gradient", fields)};
    FieldT & f_grad_test{muGrid::make_field<FieldT>("gradient_test", fields)};
    FieldMap grad(f_grad);
    FieldMap grad_test(f_grad_test);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    for (auto && tup : akantu::zip(fields, grad, grad_test)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & gt = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, muGrid::operator/(fix::projector.get_domain_lengths(),
                                    fix::projector.get_nb_domain_grid_pts()));
      g.row(0) = vec.transpose() * cos(vec.dot(vec));
      gt.row(0) = g.row(0);
    }

    fix::projector.initialise(muFFT::FFT_PlanFlags::estimate);
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
