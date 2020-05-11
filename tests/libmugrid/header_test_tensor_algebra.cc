/**
 * @file   header_test_tensor_algebra.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Tests for the tensor algebra functions
 *
 * Copyright © 2017 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include <iomanip>

#include <unsupported/Eigen/CXX11/Tensor>

#include "tests.hh"
#include "test_goodies.hh"
#include "libmugrid/eigen_tools.hh"
#include "libmugrid/tensor_algebra.hh"
#include <iostream>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(tensor_algebra)
  auto TerrNorm = [](auto && t) {
    return Eigen::Tensor<Real, 0>(t.abs().sum())();
  };

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(tensor_outer_product_test) {
    constexpr Dim_t dim{2};
    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>> A, B;
    // use prime numbers so that every multiple is uniquely identifiable
    A.setValues({{1, 2}, {3, 7}});
    B.setValues({{11, 13}, {17, 19}});

    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>> Res1, Res2,
        Res3;

    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t k = 0; k < dim; ++k) {
          for (Dim_t l = 0; l < dim; ++l) {
            Res1(i, j, k, l) = A(i, j) * B(k, l);
            Res2(i, j, k, l) = A(i, k) * B(j, l);
            Res3(i, j, k, l) = A(i, l) * B(j, k);
          }
        }
      }
    }

    Real error = TerrNorm(Res1 - Tensors::outer<dim>(A, B));
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(Res2 - Tensors::outer_under<dim>(A, B));
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(Res3 - Tensors::outer_over<dim>(A, B));
    if (error > tol) {
      std::cout << "reference:" << std::endl << Res3 << std::endl;
      std::cout << "result:" << std::endl
                << Tensors::outer_over<dim>(A, B) << std::endl;
      std::cout << "A:" << std::endl << A << std::endl;
      std::cout << "B" << std::endl << B << std::endl;
      decltype(Res3) tmp = Tensors::outer_over<dim>(A, B);
      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              std::cout << "for (" << i << ", " << j << ", " << k << ", " << l
                        << "), ref: " << std::setw(3) << Res3(i, j, k, l)
                        << ", res: " << std::setw(3) << tmp(i, j, k, l)
                        << std::endl;
            }
          }
        }
      }
    }
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(Res3 - Tensors::outer<dim>(A, B));
    BOOST_CHECK_GT(error, tol);
  };

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(outer_products, Fix, testGoodies::dimlist,
                                   Fix) {
    constexpr auto dim{Fix::dim};
    using T2 = Tensors::Tens2_t<dim>;
    using M2 = Matrices::Tens2_t<dim>;
    using Map2 = Eigen::Map<const M2>;
    using T4 = Tensors::Tens4_t<dim>;
    using M4 = Matrices::Tens4_t<dim>;
    using Map4 = Eigen::Map<const M4>;
    T2 A, B;
    T4 RT;
    A.setRandom();
    B.setRandom();
    Map2 Amap(A.data());
    Map2 Bmap(B.data());
    M2 C, D;
    M4 RM;
    C = Amap;
    D = Bmap;

    auto error = [](const T4 & A, const M4 & B) {
      return (B - Map4(A.data())).norm();
    };

    // Check outer product
    RT = Tensors::outer<dim>(A, B);
    RM = Matrices::outer(C, D);
    BOOST_CHECK_LT(error(RT, RM), tol);

    // Check outer_under product
    RT = Tensors::outer_under<dim>(A, B);
    RM = Matrices::outer_under(C, D);
    BOOST_CHECK_LT(error(RT, RM), tol);

    // Check outer_over product
    RT = Tensors::outer_over<dim>(A, B);
    RM = Matrices::outer_over(C, D);
    BOOST_CHECK_LT(error(RT, RM), tol);
  }

  BOOST_AUTO_TEST_CASE(tensor_multiplication) {
    constexpr Dim_t dim{2};
    using Strain_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>>;
    Strain_t A, B;
    A.setValues({{1, 2}, {3, 7}});
    B.setValues({{11, 13}, {17, 19}});
    Strain_t FF1 = A * B;  // element-wise multiplication
    std::array<Eigen::IndexPair<int>, 1> prod_dims{Eigen::IndexPair<int>{1, 0}};

    Strain_t FF2 = A.contract(B, prod_dims);  // correct option 1
    Strain_t FF3;
    using Mat_t = Eigen::Map<Eigen::Matrix<Real, dim, dim>>;
    // following only works for evaluated tensors (which already have data())
    Mat_t(FF3.data()) = Mat_t(A.data()) * Mat_t(B.data());
    Strain_t ref;
    ref.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t a = 0; a < dim; ++a) {
          ref(i, j) += A(i, a) * B(a, j);
        }
      }
    }

    using Strain_tw =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<dim + 1, dim + 1>>;
    Strain_tw C;
    C.setConstant(100);
    // static_assert(!std::is_convertible<Strain_t, Strain_tw>::value,
    //              "Tensors not size-protected");
    if (std::is_convertible<Strain_t, Strain_tw>::value) {
      // std::cout << "this is not good, should I abandon Tensors?";
    }
    // this test seems useless. I use to detect if Eigen changed the
    // default tensor product
    Real error = TerrNorm(FF1 - ref);
    if (error < tol) {
      std::cout << "A =" << std::endl << A << std::endl;
      std::cout << "B =" << std::endl << B << std::endl;
      std::cout << "FF1 =" << std::endl << FF1 << std::endl;
      std::cout << "ref =" << std::endl << ref << std::endl;
    }
    BOOST_CHECK_GT(error, tol);

    error = TerrNorm(FF2 - ref);
    if (error > tol) {
      std::cout << "A =" << std::endl << A << std::endl;
      std::cout << "B =" << std::endl << B << std::endl;
      std::cout << "FF2 =" << std::endl << FF2 << std::endl;
      std::cout << "ref =" << std::endl << ref << std::endl;
    }
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(FF3 - ref);
    if (error > tol) {
      std::cout << "A =" << std::endl << A << std::endl;
      std::cout << "B =" << std::endl << B << std::endl;
      std::cout << "FF3 =" << std::endl << FF3 << std::endl;
      std::cout << "ref =" << std::endl << ref << std::endl;
    }
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensmult, Fix, testGoodies::dimlist,
                                   Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T4 = Tens4_t<dim>;
    using T2 = Tens2_t<dim>;
    using V2 = Eigen::Matrix<Real, dim * dim, 1>;
    T4 C;
    C.setRandom();
    T2 E;
    E.setRandom();
    Eigen::Map<const V2> Ev(E.data());
    T2 R = tensmult(C, E);

    auto error = (Eigen::Map<const V2>(R.data()) - C * Ev).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tracer, Fix, testGoodies::dimlist,
                                   Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T2 = Tens2_t<dim>;
    auto tracer = Matrices::Itrac<dim>();
    T2 F;
    F.setRandom();
    auto Ftrac = tensmult(tracer, F);
    auto error = (Ftrac - F.trace() * F.Identity()).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_identity, Fix, testGoodies::dimlist,
                                   Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T2 = Tens2_t<dim>;
    auto ident = Matrices::Iiden<dim>();
    T2 F;
    F.setRandom();
    auto Fiden = tensmult(ident, F);
    auto error = (Fiden - F).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_transposer, Fix, testGoodies::dimlist,
                                   Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T2 = Tens2_t<dim>;
    auto trnst = Matrices::Itrns<dim>();
    T2 F;
    F.setRandom();
    auto Ftrns = tensmult(trnst, F);
    auto error = (Ftrns - F.transpose()).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_symmetriser, Fix, testGoodies::dimlist,
                                   Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T2 = Tens2_t<dim>;
    auto symmt = Matrices::Isymm<dim>();
    T2 F;
    F.setRandom();
    auto Fsymm = tensmult(symmt, F);
    auto error = (Fsymm - .5 * (F + F.transpose())).norm();
    BOOST_CHECK_LT(error, tol);
  }

  /* ---------------------------------------------------------------------- */

  template <Dim_t Dim>
  struct MatricesFixture {
    using M1_t = Eigen::Matrix<Real, Dim, 1>;
    using M2_t = Eigen::Matrix<Real, Dim, Dim>;
    using M4_t = Eigen::Matrix<Real, Dim * Dim, Dim * Dim>;
    MatricesFixture()
        : F{M2_t::Identity() + 1.0e-1 * M2_t::Random()}, m1{M1_t::Random()},
          m2{M2_t::Random()}, m4{M4_t::Random()} {}
    M2_t F;
    M1_t m1;
    M2_t m2;
    M4_t m4;
    Real tol{1e-6};
  };

  using matrices =
      boost::mpl::list<MatricesFixture<twoD>, MatricesFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_axis_transformation, Fix, matrices,
                                   Fix) {
    namespace Transform = Matrices::AxisTransform;


    auto && m1_forwarded{Transform::push_forward(Fix::m1, Fix::F)};
    auto && m2_forwarded{Transform::push_forward(Fix::m2, Fix::F)};
    auto && m4_forwarded{Transform::push_forward(Fix::m4, Fix::F)};

    auto && m1_back{Transform::pull_back(m1_forwarded, Fix::F)};
    auto && m2_back{Transform::pull_back(m2_forwarded, Fix::F)};
    auto && m4_back{Transform::pull_back(m4_forwarded, Fix::F)};

    auto && err_1{testGoodies::rel_error(Fix::m1, m1_back)};
    BOOST_CHECK_LT(err_1, Fix::tol);

    auto && err_2{testGoodies::rel_error(Fix::m2, m2_back)};
    BOOST_CHECK_LT(err_2, Fix::tol);

    auto && err_4{testGoodies::rel_error(Fix::m4, m4_back)};
    BOOST_CHECK_LT(err_4, Fix::tol);
  }
  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
