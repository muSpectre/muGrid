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

#include "tests.hh"
#include "solver/deprecated_solvers.hh"
#include "solver/deprecated_solver_cg.hh"
#include "solver/deprecated_solver_cg_eigen.hh"
#include "libmufft/fftw_engine.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "libmugrid/iterators.hh"
#include "libmugrid/ccoord_operations.hh"
#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "cell/cell_split.hh"
#include "common/intersection_octree.hh"

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(split_cell);

  BOOST_AUTO_TEST_CASE(patch_test) {
    constexpr Dim_t dim{twoD};

    using Mat_t = MaterialLinearElastic1<dim, dim>;
    const Real contrast{10};
    const Real Young_soft{1.0030648180242636},
        Poisson_soft{0.29930675909878679};
    const Real Young_hard{contrast * Young_soft},
        Poisson_hard{0.29930675909878679};
    const Real Young_mix{(Young_soft + Young_hard) / 2},
        Poisson_mix{0.29930675909878679};

    constexpr Ccoord_t<dim> nb_grid_pts_split{3, 3};
    constexpr Rcoord_t<dim> lengths_split{3, 3};
    auto fft_ptr_split{std::make_unique<muFFT::FFTWEngine<dim>>(
        nb_grid_pts_split, muGrid::ipow(dim, 2))};
    auto proj_ptr_split{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(
        std::move(fft_ptr_split), lengths_split)};
    CellSplit<dim, dim> sys_split(std::move(proj_ptr_split));

    constexpr Ccoord_t<dim> nb_grid_pts_base{3, 3};
    constexpr Rcoord_t<dim> lengths_base{3, 3};
    auto fft_ptr_base{std::make_unique<muFFT::FFTWEngine<dim>>(
        nb_grid_pts_base, muGrid::ipow(dim, 2))};
    auto proj_ptr_base{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(
        std::move(fft_ptr_base), lengths_base)};
    CellBase<dim, dim> sys_base(std::move(proj_ptr_base));

    auto & Material_hard_split{
        Mat_t::make(sys_split, "hard", Young_hard, Poisson_hard)};
    auto & Material_soft_split{
        Mat_t::make(sys_split, "soft", Young_soft, Poisson_soft)};

    for (auto && tup : akantu::enumerate(sys_split)) {
      auto && pixel{std::get<1>(tup)};
      if (pixel[0] < 2) {
        Material_hard_split.add_pixel_split(pixel, 1);
      } else {
        Material_hard_split.add_pixel_split(pixel, 0.5);
        Material_soft_split.add_pixel_split(pixel, 0.5);
      }
    }

    auto & Material_hard_base{
        Mat_t::make(sys_base, "hard", Young_hard, Poisson_hard)};
    auto & Material_mix_base{
        Mat_t::make(sys_base, "mix", Young_mix, Poisson_mix)};

    for (auto && tup : akantu::enumerate(sys_base)) {
      auto && pixel{std::get<1>(tup)};
      if (pixel[0] < 2) {
        Material_hard_base.add_pixel(pixel);
      } else {
        Material_mix_base.add_pixel(pixel);
      }
    }

    Grad_t<dim> delF0;
    delF0 << 0, 1, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5};
    constexpr Uint maxiter{muGrid::CcoordOps::get_size(nb_grid_pts_split) *
                           muGrid::ipow(dim, secondOrder) * 10};
    constexpr bool verbose{false};

    GradIncrements<dim> grads{};
    grads.push_back(delF0);

    DeprecatedSolverCG<dim> cg2{sys_base, cg_tol, maxiter,
                                static_cast<bool>(verbose)};
    Eigen::ArrayXXd res2{
        deprecated_newton_cg(sys_base, grads, cg2, newton_tol, verbose)[0]
            .grad};

    DeprecatedSolverCG<dim> cg1{sys_split, cg_tol, maxiter,
                                static_cast<bool>(verbose)};
    Eigen::ArrayXXd res1{
        deprecated_newton_cg(sys_split, grads, cg1, newton_tol, verbose)[0]
            .grad};

    BOOST_CHECK_LE(abs(res1 - res2).mean(), cg_tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
