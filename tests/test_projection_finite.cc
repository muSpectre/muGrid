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

#include <boost/mpl/list.hpp>
#include "boost/range/combine.hpp"

#include "tests.hh"
#include "fft/fftw_engine.hh"
#include "fft/projection_finite_strain.hh"
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
    constexpr static Ccoord_t<twoD> get_value() {
      return Ccoord_t<twoD>{3, 5};}
  };
  template<>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_value() {
      return Ccoord_t<threeD>{3, 5, 7};}
  };
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  struct ProjectionFixture {
    using Engine = FFTW_Engine<DimS, DimM>;
    using Parent = ProjectionFiniteStrain<DimS, DimM>;
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    ProjectionFixture(): engine{Sizes<DimS>::get_value()},
                         projector(engine){}
    Engine engine;
    Parent projector;
  };

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<ProjectionFixture<twoD, twoD>,
                                   ProjectionFixture<threeD, threeD>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(fix::projector.initialise(FFT_PlanFlags::estimate));
  }


  /* ---------------------------------------------------------------------- */
  using F3 = ProjectionFixture<threeD, threeD>;
  BOOST_FIXTURE_TEST_CASE(Helmholtz_decomposition, F3) {
    // create a field that is explicitely the sum of a gradient and a
    // curl, then verify that the projected field corresponds to the
    // gradient (without the zero'th component)
    using Fields = FieldCollection<sdim, mdim>;
    Fields fields{};
    using FieldT = TensorField<Fields, Real, secondOrder, mdim>;
    using FieldMap = MatrixFieldMap<Fields, Real, mdim, mdim>;
    FieldT & f_grad{make_field<FieldT>("gradient", fields)};
    FieldT & f_curl{make_field<FieldT>("curl", fields)};
    FieldT & f_sum{make_field<FieldT>("sum", fields)};
    FieldT & f_var{make_field<FieldT>("working field", fields)};

    FieldMap grad(f_grad);
    FieldMap curl(f_curl);
    FieldMap sum(f_sum);
    FieldMap var(f_var);

    fields.initialise(Sizes<sdim>::get_value());

    for (auto && tup: boost::combine(fields, grad, curl, sum, var)) {
      auto & ccoord = boost::get<0>(tup);
      auto & g = boost::get<1>(tup);
      auto & c = boost::get<2>(tup);
      auto & s = boost::get<3>(tup);
      auto & v = boost::get<4>(tup);
      auto vec = CcoordOps::get_vector(ccoord);
      g.row(0) << vec(1)*vec(2), vec(0)*vec(2), vec(1)*vec(2);
      c.row(0) << 0, vec(0)*vec(1), -vec(0)*vec(2);
      v.row(0) = s.row(0) = g.row(0) + c.row(0);
    }

    projector.initialise(FFT_PlanFlags::estimate);
    projector.apply_projection(f_var);

    for (auto && tup: boost::combine(fields, grad, curl, sum, var)) {
      auto & ccoord = boost::get<0>(tup);
      auto & g = boost::get<1>(tup);
      auto & c = boost::get<2>(tup);
      auto & s = boost::get<3>(tup);
      auto & v = boost::get<4>(tup);
      auto vec = CcoordOps::get_vector(ccoord);

      Real error = (s-g).norm();
      BOOST_CHECK_LT(error, tol);
      if (error >=tol) {
        std::cout << std::endl << "grad_ref :"  << std::endl << g << std::endl;
        std::cout << std::endl << "grad_proj :" << std::endl << v << std::endl;
        std::cout << std::endl << "curl :"      << std::endl << c << std::endl;
        std::cout << std::endl << "sum :"       << std::endl << s << std::endl;
        std::cout << std::endl << "ccoord :"    << std::endl << ccoord << std::endl;
        std::cout << std::endl << "vector :"    << std::endl << vec.transpose() << std::endl;
      }
    }
  }
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
