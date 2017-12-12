/**
 * file   test_projection_finite.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Dec 2017
 *
 * @brief  tests for standard finite strain projection operator
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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
#include <cmath>

#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>
#include <Eigen/Dense>

#include "tests.hh"
#include "fft/fftw_engine.hh"
#include "fft/projection_finite_strain.hh"
#include "fft/fft_utils.hh"
#include "common/common.hh"
#include "common/field_collection.hh"
#include "common/field_map.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain);


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  struct Sizes {
  };
  template<>
  struct Sizes<twoD> {
    constexpr static Ccoord_t<twoD> get_resolution() {
      return Ccoord_t<twoD>{3, 5};}
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{3.4, 5.8};}
  };
  template<>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_resolution() {
      return Ccoord_t<threeD>{3, 5, 7};}
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{3.4, 5.8, 6.7};}
  };
  template <Dim_t DimS>
  struct Squares {
  };
  template<>
  struct Squares<twoD> {
    constexpr static Ccoord_t<twoD> get_resolution() {
      return Ccoord_t<twoD>{5, 5};}
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{5, 5};}
  };
  template<>
  struct Squares<threeD> {
    constexpr static Ccoord_t<threeD> get_resolution() {
      return Ccoord_t<threeD>{7, 7, 7};}
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{7, 7, 7};}
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class SizeGiver>
  struct ProjectionFixture {
    using Engine = FFTW_Engine<DimS, DimM>;
    using Parent = ProjectionFiniteStrain<DimS, DimM>;
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    ProjectionFixture(): engine{SizeGiver::get_resolution(),
        SizeGiver::get_lengths()},
                         projector(engine){}
    Engine engine;
    Parent projector;
  };

  /* ---------------------------------------------------------------------- */
  using fixlist =
    boost::mpl::list<ProjectionFixture<twoD, twoD, Squares<twoD>>,
                     ProjectionFixture<threeD, threeD, Squares<threeD>>,
                     ProjectionFixture<twoD, twoD, Sizes<twoD>>,
                     ProjectionFixture<threeD, threeD, Sizes<threeD>>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
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
    using Fields = FieldCollection<sdim, mdim>;
    using FieldT = TensorField<Fields, Real, secondOrder, mdim>;
    using FieldMap = MatrixFieldMap<Fields, Real, mdim, mdim>;
    using Vector = Eigen::Matrix<Real, dim, 1>;

    Fields fields{};
    FieldT & f_grad{make_field<FieldT>("gradient", fields)};
    FieldT & f_var{make_field<FieldT>("working field", fields)};

    FieldMap grad(f_grad);
    FieldMap var(f_var);

    fields.initialise(fix::engine.get_resolutions());
    FFT_freqs<dim> freqs{fix::engine.get_resolutions(),
        fix::engine.get_lengths()};
    Vector k; for (Dim_t i = 0; i < dim; ++i) {
      // the wave vector has to be such that it leads to an integer
      // number of periods in each length of the domain
      k(i) = (i+1)*2*pi/fix::engine.get_lengths()[i]; ;
    }

    for (auto && tup: boost::combine(fields, grad, var)) {
      auto & ccoord = boost::get<0>(tup);
      auto & g = boost::get<1>(tup);
      auto & v = boost::get<2>(tup);
      Vector vec = CcoordOps::get_vector(ccoord,
                                         fix::engine.get_lengths()/
                                         fix::engine.get_resolutions());
      g.row(0) << k.transpose() * cos(k.dot(vec));
      v.row(0) = g.row(0);
    }

    fix::projector.initialise(FFT_PlanFlags::estimate);
    fix::projector.apply_projection(f_var);

    for (auto && tup: boost::combine(fields, grad, var)) {
      auto & ccoord = boost::get<0>(tup);
      auto & g = boost::get<1>(tup);
      auto & v = boost::get<2>(tup);
      Vector vec = CcoordOps::get_vector(ccoord,
                                         fix::engine.get_lengths()/
                                         fix::engine.get_resolutions());
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

  // /* ---------------------------------------------------------------------- */
  // using F3 = ProjectionFixture<threeD, threeD, Squares<threeD>>;
  // BOOST_FIXTURE_TEST_CASE(Helmholtz_decomposition, F3) {
  //   // create a field that is explicitely the sum of a gradient and a
  //   // curl, then verify that the projected field corresponds to the
  //   // gradient (without the zero'th component)
  //   constexpr Dim_t sdim{F3::sdim}, mdim{F3::mdim};
  //   using Fields = FieldCollection<sdim, mdim>;
  //   Fields fields{};
  //   using FieldT = TensorField<Fields, Real, secondOrder, mdim>;
  //   using FieldMap = MatrixFieldMap<Fields, Real, mdim, mdim>;
  //   FieldT & f_grad{make_field<FieldT>("gradient", fields)};
  //   FieldT & f_curl{make_field<FieldT>("curl", fields)};
  //   FieldT & f_sum{make_field<FieldT>("sum", fields)};
  //   FieldT & f_var{make_field<FieldT>("working field", fields)};

  //   FieldMap grad(f_grad);
  //   FieldMap curl(f_curl);
  //   FieldMap sum(f_sum);
  //   FieldMap var(f_var);

  //   fields.initialise(F3::engine.get_resolutions());

  //   for (auto && tup: boost::combine(fields, grad, curl, sum, var)) {
  //     auto & ccoord = boost::get<0>(tup);
  //     auto & g = boost::get<1>(tup);
  //     auto & c = boost::get<2>(tup);
  //     auto & s = boost::get<3>(tup);
  //     auto & v = boost::get<4>(tup);
  //     auto vec = CcoordOps::get_vector(ccoord);
  //     Real & x{vec(0)};
  //     Real & y{vec(1)};
  //     Real & z{vec(2)};
  //     g.row(0) << y*z, x*z, y*z;
  //     c.row(0) << 0, x*y, -x*z;
  //     v.row(0) = s.row(0) = g.row(0) + c.row(0);
  //   }

  //   projector.initialise(FFT_PlanFlags::estimate);
  //   projector.apply_projection(f_var);

  //   for (auto && tup: boost::combine(fields, grad, curl, sum, var)) {
  //     auto & ccoord = boost::get<0>(tup);
  //     auto & g = boost::get<1>(tup);
  //     auto & c = boost::get<2>(tup);
  //     auto & s = boost::get<3>(tup);
  //     auto & v = boost::get<4>(tup);
  //     auto vec = CcoordOps::get_vector(ccoord);

  //     Real error = (s-g).norm();
  //     //BOOST_CHECK_LT(error, tol);
  //     if (error >=tol) {
  //       std::cout << std::endl << "grad_ref :"  << std::endl << g << std::endl;
  //       std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
  //       std::cout << std::endl << "curl :"      << std::endl << c << std::endl;
  //       std::cout << std::endl << "sum :"       << std::endl << s << std::endl;
  //       std::cout << std::endl << "ccoord :"    << std::endl << ccoord << std::endl;
  //       std::cout << std::endl << "vector :"    << std::endl << vec.transpose() << std::endl;
  //     }
  //   }
  // }
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
