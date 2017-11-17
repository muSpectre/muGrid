/**
 * file   test_tensor_algebra.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Tests for the tensor algebra functions
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

#include <iomanip>

#include <unsupported/Eigen/CXX11/Tensor>

#include "common/tensor_algebra.hh"
#include "tests.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(tensor_algebra)
  auto TerrNorm = [](auto && t){
    return Eigen::Tensor<Real, 0>(t.abs().sum())();
  };

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(outer_product_test) {
    constexpr Dim_t dim{2};
    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim,dim>> A, B;
    // use prime numbers so that every multiple is uniquely identifiable
    A.setValues({{1, 2}, {3 ,7}});
    B.setValues({{11, 13}, {17 ,19}});

    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>> Res1, Res2, Res3;

    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t k = 0; k < dim; ++k) {
          for (Dim_t l = 0; l < dim; ++l) {
            Res1(i, j, k, l) = A(i, j)*B(k, l);
            Res2(i, j, k, l) = A(i, k)*B(j, l);
            Res3(i, j, k, l) = A(i, l)*B(j, k);
          }
        }
      }
    }


    Real error = TerrNorm(Res1 - Tensors::outer<dim>(A,B));
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(Res2 - Tensors::outer_under<dim>(A, B));
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(Res3 - Tensors::outer_over<dim>(A, B));
    if (error > tol) {
      std::cout << "reference:" << std::endl
                << Res3 << std::endl;
      std::cout << "result:" << std::endl
                << Tensors::outer_over<dim>(A, B) << std::endl;
      std::cout << "A:" << std::endl
                << A << std::endl;
      std::cout << "B" << std::endl
                << B << std::endl;
      decltype(Res3) tmp = Tensors::outer_over<dim>(A, B);
      for (Dim_t i = 0; i < dim; ++i) {
        for (Dim_t j = 0; j < dim; ++j) {
          for (Dim_t k = 0; k < dim; ++k) {
            for (Dim_t l = 0; l < dim; ++l) {
              std::cout << "for (" << i << ", " << j << ", " << k << ", " << l
                        << "), ref: " << std::setw(3)<< Res3(i,j,k,l)
                        << ", res: " <<  std::setw(3)<< tmp(i,j,k,l)
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

  BOOST_AUTO_TEST_CASE(tensor_multiplication) {
    constexpr Dim_t dim{2};
    using Strain_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>>;
    Strain_t A, B;
    A.setValues({{1, 2}, {3 ,7}});
    B.setValues({{11, 13}, {17 ,19}});
    Strain_t FF1 = A*B; // element-wise multiplication
    std::array<Eigen::IndexPair<int>, 1> prod_dims
    {   Eigen::IndexPair<int>{1, 0}};

    Strain_t FF2 = A.contract(B, prod_dims); // correct option 1
    Strain_t FF3;
    using Mat_t = Eigen::Map<Eigen::Matrix<Real, dim, dim>>;
    // following only works for evaluated tensors (which already have data())
    Mat_t(FF3.data()) = Mat_t(A.data()) * Mat_t(B.data());
    Strain_t ref;
    ref.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t a = 0; a < dim; ++a) {
          ref(i,j) += A(i, a) * B(a, j);
        }
      }
    }


    using T = Eigen::TensorFixedSize<Real, Eigen::Sizes<2, 2> >;
    constexpr Dim_t my_size =
      T::NumIndices;
    std::cout << "size = " << my_size << std::endl;

    using Strain_tw = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim+1, dim+1>>;
    Strain_tw C;
    C.setConstant(100);
    auto a = T::Dimensions::total_size;
    auto b = Strain_tw::Dimensions::total_size;
    std::cout << "a, b = " << a << ", " << b << std::endl;
    //static_assert(!std::is_convertible<Strain_t, Strain_tw>::value,
    //              "Tensors not size-protected");
    if (std::is_convertible<Strain_t, Strain_tw>::value) {
      std::cout << "this is not good, should I abandon Tensors?";
    }
    // this test seems useless. I use to detect if Eigen changed the
    // default tensor product
    Real error = TerrNorm(FF1-ref);
    if (error < tol) {
      std::cout << "A =" << std::endl
                << A << std::endl;
      std::cout << "B =" << std::endl
                << B << std::endl;
      std::cout << "FF1 =" << std::endl
                << FF1 << std::endl;
      std::cout << "ref =" << std::endl
                << ref << std::endl;
    }
    BOOST_CHECK_GT(error, tol);

    error = TerrNorm(FF2-ref);
    if (error > tol) {
      std::cout << "A =" << std::endl
                << A << std::endl;
      std::cout << "B =" << std::endl
                << B << std::endl;
      std::cout << "FF2 =" << std::endl
                << FF2 << std::endl;
      std::cout << "ref =" << std::endl
                << ref << std::endl;
    }
    BOOST_CHECK_LT(error, tol);

    error = TerrNorm(FF3-ref);
    if (error > tol) {
      std::cout << "A =" << std::endl
                << A << std::endl;
      std::cout << "B =" << std::endl
                << B << std::endl;
      std::cout << "FF3 =" << std::endl
                << FF3 << std::endl;
      std::cout << "ref =" << std::endl
                << ref << std::endl;
    }
    BOOST_CHECK_LT(error, tol);

  }
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
