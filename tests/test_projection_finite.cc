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
#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "test_projection.hh"

#include <libmufft/fft_utils.hh>
#include <libmugrid/field_typed.hh>

#include <Eigen/Dense>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
      ProjectionFixture<twoD, twoD, Squares<twoD>, FourierGradient<twoD>,
                        ProjectionFiniteStrain<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrain<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>, FourierGradient<twoD>,
                        ProjectionFiniteStrain<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>, FourierGradient<threeD>,
                        ProjectionFiniteStrain<threeD>>,

      ProjectionFixture<twoD, twoD, Squares<twoD>, FourierGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        FourierGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>, FourierGradient<twoD>,
                        ProjectionFiniteStrainFast<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>, FourierGradient<threeD>,
                        ProjectionFiniteStrainFast<threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(
        fix::projector.initialise());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(even_grid_test) {
    using Engine = muFFT::FFTWEngine;
    using proj = ProjectionFiniteStrainFast<twoD>;
    auto nb_dof{2 * 2};
    auto engine = std::make_unique<Engine>(DynCcoord_t{2, 3});
    engine->create_plan(nb_dof);
    BOOST_CHECK_THROW(proj(std::move(engine), DynRcoord_t{4.3, 4.3}),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test, fix, fixlist,
                                   fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(
        dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = muGrid::GlobalFieldCollection;
    using FieldMap = muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim,
                                            IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    muGrid::RealField & f_grad{fields.register_real_field(
        "gradient", mdim * mdim, QuadPtTag)};
    muGrid::RealField & f_var{fields.register_real_field(
        "working field", mdim * mdim, QuadPtTag)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fix::projector.initialise();

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());
    fix::projector.apply_projection(f_var);

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
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(idempotent_test, fix, fixlist, fix) {
    // check if the exact projection operator is a valid projection operator.
    // Thus it has to be idempotent, G^2=G or G:G:test_field = G:test_field.
    constexpr Dim_t sdim{fix::sdim}, mdim{fix::mdim};
    using Fields = muGrid::GlobalFieldCollection;
    using FieldT = muGrid::TypedField<Real>;
    using FieldMap = muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim,
                                            IterUnit::SubPt>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    FieldT & f_grad{fields.register_real_field("gradient", mdim * mdim,
                                               QuadPtTag)};
    FieldT & f_grad_test{fields.register_real_field(
        "gradient_test", mdim * mdim, QuadPtTag)};
    FieldMap grad(f_grad);
    FieldMap grad_test(f_grad_test);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
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
