/**
 * @file   test_eigen_strain_solver.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   27 May 2020
 *
 * @brief  Testing the eigen strain handling of solvers against eigen strain
 * handling of materials
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
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "cell/cell_factory.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmufft/fftw_engine.hh>

#include <boost/mpl/list.hpp>
#include <functional>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(eigen_strain_hendling_solvers);

  BOOST_AUTO_TEST_CASE(twoD_test) {
    constexpr Index_t Dim{twoD};

    using Mat1_t = MaterialLinearElastic1<Dim>;
    using Mat2_t = MaterialLinearElastic2<Dim>;
    using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;

    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    auto cell_solver{make_cell(nb_grid_pts, lengths, form)};
    auto cell_material{make_cell(nb_grid_pts, lengths, form)};

    constexpr Real Young{2.}, Poisson{.33};

    auto & material_1_material{
        Mat1_t::make(cell_material, "material_1_material", Young, Poisson)};
    auto & material_2_material{
        Mat2_t::make(cell_material, "material_2_material", Young, Poisson)};
    auto & material_1_solver{
        Mat1_t::make(cell_solver, "material_1_solver", Young, Poisson)};

    for (const auto && index_pixel : cell_solver.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      material_1_solver.add_pixel(index);
    }

    Matrix_t F_eigen{
        (Matrix_t() << 1.43e-4, 1.1413e-4, 1.1413e-4, 6.2432e-5).finished()};

    for (const auto && index_pixel : cell_material.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      if (index == Index_t(cell_material.get_nb_pixels() / 2)) {
        material_2_material.add_pixel(index, F_eigen);
      } else {
        material_1_material.add_pixel(index);
      }
    }

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func{
        [&cell_material, &F_eigen](const size_t & /*step*/,
                                   muGrid::TypedFieldBase<Real> & eval_field) {
          auto shape{cell_material.get_strain_shape()};
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
                eval_field, shape[0], muGrid::IterUnit::SubPt)};
          for (auto && tup : eigen_field_map.enumerate_indices()) {
            auto && index{std::get<0>(tup)};
            auto && eigen{std::get<1>(tup)};
            if (index == cell_material.get_nb_pixels() / 2) {
              eigen -= F_eigen;
            }
          }
        }};

    cell_material.initialise();
    cell_solver.initialise();

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{0};
    constexpr Dim_t maxiter{100};
    constexpr Verbosity verbose{Verbosity::Silent};

    KrylovSolverCGEigen cg_material{cell_material, cg_tol, maxiter, verbose};
    KrylovSolverCGEigen cg_solver{cell_solver, cg_tol, maxiter, verbose};

    Matrix_t delF0{(Matrix_t() << 2.43e-3, 1e-3, 1e-3, 0).finished()};

    auto && res_material{newton_cg(cell_material, delF0, cg_material,
                                   newton_tol, equi_tol, verbose,
                                   IsStrainInitialised::False)};

    auto && res_solver{newton_cg(cell_solver, delF0, cg_solver, newton_tol,
                                 equi_tol, verbose, IsStrainInitialised::False,
                                 eigen_func)};

    auto && stress_solver{res_solver.stress};
    auto && stress_material{res_material.stress};

    auto && strain_solver{res_solver.grad};
    auto && strain_material{res_material.grad};

    auto && diff_stress{rel_error(stress_solver, stress_material)};
    auto && diff_strain{rel_error(strain_solver, strain_material)};

    BOOST_CHECK_LE(diff_stress, cg_tol);
    BOOST_CHECK_LE(diff_strain, cg_tol);
    std::cout << "I am here" << std::endl;
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
