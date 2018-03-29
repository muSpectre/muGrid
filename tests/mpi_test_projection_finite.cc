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

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 50

#include "fft/projection_finite_strain.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/fft_utils.hh"
#include "mpi_test_projection.hh"

#include "fft/fftw_engine.hh"
#ifdef WITH_FFTWMPI
#include "fft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "fft/pfft_engine.hh"
#endif

#include <Eigen/Dense>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(mpi_projection_finite_strain);

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<
#ifdef WITH_FFTWMPI
    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>,
                      FFTWMPIEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>,
                      FFTWMPIEngine<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>,
                      FFTWMPIEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>,
                      FFTWMPIEngine<threeD, threeD>>,

    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>,
                      FFTWMPIEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>,
                      FFTWMPIEngine<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>,
                      FFTWMPIEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>,
                      FFTWMPIEngine<threeD, threeD>>,
#endif
#ifdef WITH_PFFT
    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>,
                      PFFTEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>,
                      PFFTEngine<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>,
                      PFFTEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrain<threeD, threeD>,
                      PFFTEngine<threeD, threeD>>,

    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>,
                      PFFTEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Squares<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>,
                      PFFTEngine<threeD, threeD>>,
    ProjectionFixture<twoD, twoD, Sizes<twoD>,
                      ProjectionFiniteStrainFast<twoD, twoD>,
                      PFFTEngine<twoD, twoD>>,
    ProjectionFixture<threeD, threeD, Sizes<threeD>,
                      ProjectionFiniteStrainFast<threeD, threeD>,
                      PFFTEngine<threeD, threeD>>,
#endif
    ProjectionFixture<twoD, twoD, Squares<twoD>,
                      ProjectionFiniteStrain<twoD, twoD>,
                      FFTWEngine<twoD, twoD>,
                      false>
  >;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    if (fix::is_parallel || fix::projector.get_communicator().size() == 1) {
      BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
    }
    else {
      BOOST_CHECK_THROW(fix::projector.initialise(FFT_PlanFlags::estimate),
                        std::runtime_error);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Gradient_preservation_test,
                                   fix, fixlist, fix) {
    if (!fix::is_parallel || fix::projector.get_communicator().size() > 1) {
      return;
    }
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

    fields.initialise(fix::projector.get_resolutions(),
                      fix::projector.get_locations());
    Vector k; for (Dim_t i = 0; i < dim; ++i) {
      // the wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i+1)*2*pi/fix::projector.get_lengths()[i];
    }

    for (auto && tup: akantu::zip(fields, grad, var)) {
      auto & ccoord = std::get<0>(tup);
      auto & g = std::get<1>(tup);
      auto & v = std::get<2>(tup);
      Vector vec = CcoordOps::get_vector(ccoord,
                                         fix::projector.get_lengths()/
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
      Real error = (g-v).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >=tol) {
        Vector vec = CcoordOps::get_vector(ccoord,
                                           fix::projector.get_lengths()/
                                           fix::projector.get_domain_resolutions());
        std::cout << std::endl << "grad_ref :"  << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "ccoord :"    << std::endl << ccoord << std::endl;
        std::cout << std::endl << "vector :"    << std::endl << vec.transpose() << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
