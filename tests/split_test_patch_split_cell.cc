/**
 * @file   split_test_patch_split_cell.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Tests for the split cells
 *
 * Copyright © 2017 Till Junge
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

#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "cell/cell_factory.hh"
#include "cell/cell_split.hh"
#include "common/intersection_octree.hh"

#include <libmufft/fftw_engine.hh>

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(split_cell);

  BOOST_AUTO_TEST_CASE(patch_test) {
    constexpr Dim_t dim{twoD};

    using Mat_t = MaterialLinearElastic1<dim>;
    const Real contrast{10};
    const Real Poisson{0.3};
    const Real Young_soft{1.0};
    const Real Young_hard{contrast * Young_soft};
    const Real Young_mix{(Young_soft + Young_hard) / 2};

    DynCcoord_t nb_grid_pts_split{3, 3};
    DynRcoord_t lengths_split{3, 3};
    auto && fft_ptr_split{
        std::make_unique<muFFT::FFTWEngine>(nb_grid_pts_split)};
    auto && proj_ptr_split{std::make_unique<ProjectionFiniteStrainFast<dim>>(
        std::move(fft_ptr_split), lengths_split)};
    CellSplit sys_split(std::move(proj_ptr_split));

    DynCcoord_t nb_grid_pts_base{3, 3};
    DynRcoord_t lengths_base{3, 3};
    auto && fft_ptr_base{std::make_unique<muFFT::FFTWEngine>(nb_grid_pts_base)};
    auto && proj_ptr_base{std::make_unique<ProjectionFiniteStrainFast<dim>>(
        std::move(fft_ptr_base), lengths_base)};
    Cell sys_base(std::move(proj_ptr_base));

    auto & Material_hard_split{
        Mat_t::make(sys_split, "hard", Young_hard, Poisson)};
    auto & Material_soft_split{
        Mat_t::make(sys_split, "soft", Young_soft, Poisson)};

    for (auto && tup : sys_split.get_pixels().enumerate()) {
      auto && pixel_id{std::get<0>(tup)};
      auto && pixel{std::get<1>(tup)};
      if (pixel[0] < 2) {
        Material_hard_split.add_pixel_split(pixel_id, 1.0);
      } else {
        Material_hard_split.add_pixel_split(pixel_id, 0.5);
        Material_soft_split.add_pixel_split(pixel_id, 0.5);
      }
    }
    Material_hard_split.initialise();
    Material_soft_split.initialise();

    auto & Material_hard_base{
        Mat_t::make(sys_base, "hard", Young_hard, Poisson)};
    auto & Material_mix_base{Mat_t::make(sys_base, "mix", Young_mix, Poisson)};

    for (auto && tup : sys_base.get_pixels().enumerate()) {
      auto && pixel_id{std::get<0>(tup)};
      auto && pixel{std::get<1>(tup)};
      if (pixel[0] < 2) {
        Material_hard_base.add_pixel(pixel_id);
      } else {
        Material_mix_base.add_pixel(pixel_id);
      }
    }

    Grad_t<dim> delF0{Grad_t<dim>::Zero()};
    delF0 << 0, 1, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{0};
    auto && maxiter{
        (unsigned int)muGrid::CcoordOps::get_size(nb_grid_pts_split) *
        muGrid::ipow(dim, secondOrder) * 10};
    constexpr Verbosity verbose{Verbosity::Silent};

    KrylovSolverCG cg2{sys_base, cg_tol, maxiter, verbose};
    auto && res2{newton_cg(sys_base, delF0, cg2, newton_tol, equi_tol,
                           verbose).grad};

    KrylovSolverCG cg1{sys_split, cg_tol, maxiter, verbose};
    auto && res1{newton_cg(sys_split, delF0, cg1, newton_tol, equi_tol,
                           verbose).grad};
    auto && diff{(res1 - res2).eval()};

    BOOST_CHECK_LE(abs(diff).mean(), cg_tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
