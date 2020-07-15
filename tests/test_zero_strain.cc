/**
 * @file   test_zero_strain.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   11 Jun 2020
 *
 * @brief  Test solving a cell with zero strain
 *
 * Copyright © 2020 Ali Falsafi
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

#include "solver/solvers.hh"
#include "solver/krylov_solver_eigen.hh"
#include "solver/krylov_solver_cg.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "cell/cell_factory.hh"

#include <libmugrid/ccoord_operations.hh>
#include <libmufft/fftw_engine.hh>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(zero_strain);
  template <class Solver>
  struct SolverFixture {
    using type = Solver;
  };

  using solvers = boost::mpl::list<SolverFixture<KrylovSolverCGEigen>,
                                   SolverFixture<KrylovSolverCG>>;

  using solver_cg = boost::mpl::list<SolverFixture<KrylovSolverCG>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_non_zero_equil_tol, fix, solvers, fix) {
    constexpr Index_t Dim{twoD};

    using Mat1_t = MaterialLinearElastic1<Dim>;
    using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;

    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    auto cell_material{make_cell(nb_grid_pts, lengths, form)};
    constexpr Real Young{2.}, Poisson{.33};

    auto & material_1{
        Mat1_t::make(cell_material, "material_1_material", Young, Poisson)};

    for (const auto && index_pixel : cell_material.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      material_1.add_pixel(index);
    }

    cell_material.initialise();

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{1e-8};
    constexpr Dim_t maxiter{100};
    constexpr Verbosity verbose{Verbosity::Some};

    typename fix::type cg_material{cell_material, cg_tol, maxiter, verbose};

    Matrix_t delF0{Matrix_t::Zero()};

    auto && res{newton_cg(cell_material, delF0, cg_material, newton_tol,
                          equi_tol, verbose, IsStrainInitialised::False)};
    BOOST_CHECK(res.success);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_zero_equil_tol, fix, solver_cg, fix) {
    constexpr Index_t Dim{twoD};

    using Mat1_t = MaterialLinearElastic1<Dim>;
    using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;

    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    auto cell_material{make_cell(nb_grid_pts, lengths, form)};
    constexpr Real Young{2.}, Poisson{.33};

    auto & material_1{
        Mat1_t::make(cell_material, "material_1_material", Young, Poisson)};

    for (const auto && index_pixel : cell_material.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      material_1.add_pixel(index);
    }

    cell_material.initialise();

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{0};
    constexpr Dim_t maxiter{100};
    constexpr Verbosity verbose{Verbosity::Some};

    typename fix::type cg_material{cell_material, cg_tol, maxiter, verbose};

    Matrix_t delF0{Matrix_t::Zero()};

    auto && counter_before{cg_material.get_counter()};
    auto && res{newton_cg(cell_material, delF0, cg_material, newton_tol,
                          equi_tol, verbose, IsStrainInitialised::False)};
    auto && counter_after{cg_material.get_counter()};
    BOOST_CHECK(res.success);
    BOOST_CHECK_EQUAL(counter_before, counter_after);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
