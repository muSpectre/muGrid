/**
 * @file   test_projection_small.cc
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

#include "test_projection.hh"
#include "libmugrid/test_goodies.hh"

#include "projection/projection_small_strain.hh"
#include "projection/projection_approx_Green_operator.hh"

#include <libmufft/fft_utils.hh>

#include <Eigen/Dense>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_small_strain);

  using fixlist = boost::mpl::list<
      ProjectionFixture<twoD, twoD, Squares<twoD>, FourierGradient<twoD>,
                        ProjectionSmallStrain<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        FourierGradient<threeD>, ProjectionSmallStrain<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>, FourierGradient<twoD>,
                        ProjectionSmallStrain<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>, FourierGradient<threeD>,
                        ProjectionSmallStrain<threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(
        fix::projector.initialise());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(green_test, fix, fixlist, fix) {
    fix::projector.initialise();
    // create a C_ref function as 4T symetrisation
    auto && C_ref(muGrid::Matrices::Isymm<fix::sdim>());
    // create a Green operator projector

    auto && fft_pointer{std::make_unique<muFFT::FFTWEngine>(
        DynCcoord_t(fix::SizeGiver::get_nb_grid_pts()))};

    ProjectionApproxGreenOperator<fix::mdim> green_projection{
        std::move(fft_pointer), DynRcoord_t(fix::SizeGiver::get_lengths()),
        C_ref};
    green_projection.initialise();

    Real error{muGrid::testGoodies::rel_error(fix::projector.get_operator(),
                                              green_projection.get_operator())};

    BOOST_CHECK_LT(error, tol);
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
    using FieldT = muGrid::RealField;
    using FieldMap = muGrid::MatrixFieldMap<Real, Mapping::Mut, mdim, mdim,
                                            IterUnit::SubPt>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{sdim};
    fields.set_nb_sub_pts(QuadPtTag, OneQuadPt);
    FieldT & f_grad{
        fields.register_real_field("strain", mdim * mdim, QuadPtTag)};
    FieldT & f_var{
        fields.register_real_field("working field", mdim * mdim, QuadPtTag)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());

    muFFT::FFT_freqs<dim> freqs{fix::projector.get_nb_domain_grid_pts(),
                                fix::projector.get_domain_lengths()};

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
          ccoord.template get<mdim>(), (fix::projector.get_domain_lengths() /
                                        fix::projector.get_nb_domain_grid_pts())
                                           .template get<mdim>());
      g.row(0) << k.transpose() * cos(k.dot(vec));

      // We need to add I to the term, because this field has a net
      // zero gradient, which leads to a net -I strain
      g = 0.5 * ((g - g.Identity()).transpose() + (g - g.Identity())).eval() +
          g.Identity();
      v = g;
    }

    fix::projector.initialise();
    fix::projector.apply_projection(f_var);

    using muGrid::operator/;
    constexpr bool verbose{false};
    for (auto && tup : akantu::zip(
             fields.get_pixels().template get_dimensioned_pixels<mdim>(), grad,
             var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, (fix::projector.get_domain_lengths() /
                   fix::projector.get_nb_domain_grid_pts())
                      .template get<mdim>());
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if ((error >= tol) || verbose) {
        std::cout << std::endl << "grad_ref :" << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "ccoord :" << std::endl;
        muGrid::operator<<(std::cout, ccoord) << std::endl;
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
