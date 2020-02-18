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
  using muSpectre::ProjectionSmallStrain;
  using muSpectre::tol;
  using muGrid::operator/;
  using muGrid::operator<<;

  BOOST_AUTO_TEST_SUITE(mpi_projection_small_strain);

  using fixlist = boost::mpl::list<
#ifdef WITH_FFTWMPI
      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        ProjectionSmallStrain<twoD>, FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionSmallStrain<threeD>,
                        FFTWMPIEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>,
                        ProjectionSmallStrain<twoD>, FFTWMPIEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionSmallStrain<threeD>,
                        FFTWMPIEngine<threeD>>,
#endif
#ifdef WITH_PFFT
      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        ProjectionSmallStrain<twoD>, PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Squares<threeD>,
                        ProjectionSmallStrain<threeD>,
                        PFFTEngine<threeD>>,
      ProjectionFixture<twoD, twoD, Sizes<twoD>,
                        ProjectionSmallStrain<twoD>, PFFTEngine<twoD>>,
      ProjectionFixture<threeD, threeD, Sizes<threeD>,
                        ProjectionSmallStrain<threeD>,
                        PFFTEngine<threeD>>,
#endif
      ProjectionFixture<twoD, twoD, Squares<twoD>,
                        ProjectionSmallStrain<twoD>, FFTWEngine<twoD>,
                        false>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    if (fix::is_parallel || fix::projector.get_communicator().size() == 1) {
      BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
    }
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
    using Fields = muGrid::GlobalFFieldCollection<sdim>;
    using FieldT = muGrid::RealField;
    using FieldMap = muGrid::MatrixFieldMap<Real, false, mdim, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{1};
    FieldT & f_grad{fields.template register_field<FieldT>(
        "strain", mdim*mdim)};
    FieldT & f_var{fields.template register_field<FieldT>(
        "working field", mdim*mdim)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::projector.get_nb_subdomain_grid_pts(),
                      fix::projector.get_subdomain_locations());
    grad.initialise();
    var.initialise();

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
          ccoord, fix::projector.get_domain_lengths() /
                      fix::projector.get_nb_domain_grid_pts());
      g.row(0) << k.transpose() * cos(k.dot(vec));

      // We need to add I to the term, because this field has a net
      // zero gradient, which leads to a net -I strain
      g = 0.5 * ((g - g.Identity()).transpose() + (g - g.Identity())).eval() +
          g.Identity();
      v = g;
    }

    fix::projector.initialise(FFT_PlanFlags::estimate);
    fix::projector.apply_projection(f_var);

    constexpr Verbosity verbose{Verbosity::Silent};
    for (auto && tup : akantu::zip(fields.get_pixels(), grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = muGrid::CcoordOps::get_vector(
          ccoord, fix::projector.get_domain_lengths() /
                      fix::projector.get_nb_domain_grid_pts());
      Real error = (g - v).norm();
      BOOST_CHECK_LT(error, tol);
      if ((error >= tol) || verbose > Verbosity::Silent) {
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

}  // namespace muFFT
