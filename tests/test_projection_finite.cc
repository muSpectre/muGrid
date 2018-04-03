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
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#include "fft/projection_finite_strain.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/fft_utils.hh"
#include "test_projection.hh"

#include <Eigen/Dense>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>>,

    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(even_grid_test) {
    using Engine = FFTWEngine<twoD, twoD>;
    using proj = ProjectionFiniteStrainFast<twoD, twoD>;
    auto engine = std::make_unique<Engine>(Ccoord_t<twoD>{2, 2});
    BOOST_CHECK_THROW(proj(std::move(engine), Rcoord_t<twoD>{4.3, 4.3}),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test,
                                   fix, fixlist, fix) {
    // create a gradient field with a zero mean gradient and verify
    // that the projection preserves it
    constexpr Dim_t dim{fix::sdim}, sdim{fix::sdim}, mdim{fix::mdim};
    static_assert(dim == fix::mdim,
        "These tests assume that the material and spatial dimension are "
        "identical");
    using Fields = GlobalFieldCollection<sdim>;
    using FieldT = TensorField<Fields, Real, secondOrder, mdim>;
    using FieldMap = MatrixFieldMap<Fields, Real, mdim, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{};
    FieldT & f_grad{make_field<FieldT>("gradient", fields)};
    FieldT & f_var{make_field<FieldT>("working field", fields)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_subdomain_resolutions(),
                      fix::projector.get_subdomain_locations());
    FFT_freqs<dim> freqs{fix::projector.get_domain_resolutions(),
        fix::projector.get_domain_lengths()};
    Vector k; for (Dim_t i = 0; i < dim; ++i) {
      // the wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i+1)*2*pi/fix::projector.get_domain_lengths()[i]; ;
    }

    for (auto && tup: akantu::zip(fields, grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = CcoordOps::get_vector(ccoord,
                                         fix::projector.get_domain_lengths()/
                                         fix::projector.get_domain_resolutions());
      g.row(0) = k.transpose() * cos(k.dot(vec));
      v.row(0) = g.row(0);
    }

    fix::projector.initialise(FFT_PlanFlags::estimate);
    fix::projector.apply_projection(f_var);

    for (auto && tup: akantu::zip(fields, grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = CcoordOps::get_vector(ccoord,
                                         fix::projector.get_domain_lengths()/
                                         fix::projector.get_domain_resolutions());
      Real error = (g-v).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >=tol) {
        std::cout << std::endl << "grad_ref :"  << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "ccoord :"    << std::endl << ccoord << std::endl;
        std::cout << std::endl << "vector :"    << std::endl << vec.transpose() << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
