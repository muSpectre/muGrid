/**
 * @file   test_cell_data.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jul 2020
 *
 * @brief  Fixtures for CellData
 *
 * Copyright © 2020 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include <cell/cell_data.hh>

#include <boost/mpl/list.hpp>

#ifndef TESTS_TEST_CELL_DATA_HH_
#define TESTS_TEST_CELL_DATA_HH_

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct CellDataFixture {
    constexpr static Index_t SpatialDim{Dim};
    static DynCcoord_t get_size() {
      switch (SpatialDim) {
      case twoD: {
        return {3, 5};
        break;
      }
      case threeD: {
        return {3, 5, 7};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }

    static DynRcoord_t get_length() {
      switch (SpatialDim) {
      case twoD: {
        return {1, 2};
        break;
      }
      case threeD: {
        return {1, 2, 3};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }
    CellDataFixture() : cell_data(CellData::make(get_size(), get_length())) {}

    CellData_ptr cell_data;
  };

  template <Index_t Dim>
  constexpr Index_t CellDataFixture<Dim>::SpatialDim;
  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct CellDataFixtureSmall {
    constexpr static Index_t SpatialDim{Dim};
    static DynCcoord_t get_size() {
      switch (SpatialDim) {
      case twoD: {
        return {3, 3};
        break;
      }
      case threeD: {
        return {3, 3, 3};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }

    static DynRcoord_t get_length() {
      switch (SpatialDim) {
      case twoD: {
        return {1, 1};
        break;
      }
      case threeD: {
        return {1, 1, 1};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }
    CellDataFixtureSmall()
        : cell_data(CellData::make(get_size(), get_length())) {}

    CellData_ptr cell_data;
  };

  /* ---------------------------------------------------------------------- */
  using CellDataFixtures =
      boost::mpl::list<CellDataFixture<twoD>, CellDataFixture<threeD>>;

  using CellDataFixturesSmall = boost::mpl::list<CellDataFixtureSmall<twoD>,
                                                 CellDataFixtureSmall<threeD>>;

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct CellDataFixtureEigenStrain {
    constexpr static Index_t SpatialDim{Dim};
    using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;

    Matrix_t F_eigen_maker() {
      const Real eps{1.0e-6};
      switch (Dim) {
      case twoD: {
        return (Matrix_t() << eps, 0.0, 0.0, eps).finished();
        break;
      }
      case threeD: {
        return (Matrix_t() << eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, 0.0, eps)
            .finished();
        break;
      }
      default:
        throw muGrid::RuntimeError("The dimension is invalid");
        break;
      }
    }

    static DynCcoord_t get_size() {
      switch (SpatialDim) {
      case twoD: {
        return {3, 5};
        break;
      }
      case threeD: {
        return {3, 5, 7};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }

    static DynRcoord_t get_length() {
      switch (SpatialDim) {
      case twoD: {
        return {1, 2};
        break;
      }
      case threeD: {
        return {1, 2, 3};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }
    CellDataFixtureEigenStrain()
        : cell_data(CellData::make(get_size(), get_length())),
          cell_data_eigen(CellData::make(get_size(), get_length())),
          F_eigen_holder{std::make_unique<Matrix_t>(F_eigen_maker())},
          F_eigen{*F_eigen_holder}, F_0_holder{std::make_unique<Matrix_t>(
                                        Matrix_t::Zero())},
          F_0{*F_0_holder}, step_nb{0} {}

    CellData_ptr cell_data;
    CellData_ptr cell_data_eigen;

    std::unique_ptr<const Matrix_t> F_eigen_holder;  //!< eigen_strain tensor
    const Matrix_t & F_eigen;  //!< ref to eigen strain tensor

    std::unique_ptr<const Matrix_t> F_0_holder;  //!< eigen_strain tensor
    const Matrix_t & F_0;                        //!< ref to eigen strain tensor

    size_t step_nb;
  };

  /* ---------------------------------------------------------------------- */
  using CellDataFixtureEigenStrains =
      boost::mpl::list<CellDataFixtureEigenStrain<twoD>,
                       CellDataFixtureEigenStrain<threeD>>;

  using CellDataFixtureEigenStrains2D =
      boost::mpl::list<CellDataFixtureEigenStrain<twoD>>;

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct CellDataFixtureSquare {
    constexpr static Index_t SpatialDim{Dim};
    static DynCcoord_t get_size() {
      switch (SpatialDim) {
      case twoD: {
        return {3, 3};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }

    static DynRcoord_t get_length() {
      switch (SpatialDim) {
      case twoD: {
        return {1, 1};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }
    CellDataFixtureSquare()
        : cell_data(CellData::make(get_size(), get_length())) {}

    CellData_ptr cell_data;
  };

  template <Index_t Dim>
  constexpr Index_t CellDataFixtureSquare<Dim>::SpatialDim;

  /* ---------------------------------------------------------------------- */
  using CellDataFixtureSquares = boost::mpl::list<CellDataFixtureSquare<twoD>>;

}  // namespace muSpectre

#endif  // TESTS_TEST_CELL_DATA_HH_
