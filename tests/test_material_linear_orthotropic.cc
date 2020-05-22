/**
 * @file   test_material_linear_orthotropic.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Jul 2018
 *
 * @brief  Tests for testing MaterialLinearanisotropic class. (It should be
 * noted that MaterialLinearAnisotropic's evaluate stress and constructors are
 * called within calling those of Materiallinearorthotropic)
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

#include "tests.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "libmufft/fftw_engine.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_orthotropic.hh"
#include "libmugrid/iterators.hh"
#include "libmugrid/ccoord_operations.hh"
#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(orthotropic_material_tests);

  BOOST_AUTO_TEST_CASE(orthotropic_twoD) {
    constexpr Dim_t Dim{twoD};
    const DynCcoord_t resolutions{5, 5};
    const DynRcoord_t lengths{5, 5};
    auto fft_ptr_non{
        std::make_unique<muFFT::FFTWEngine>(resolutions)};

    auto proj_ptr_non{std::make_unique<ProjectionFiniteStrainFast<Dim>>(
        std::move(fft_ptr_non), lengths)};
    Cell sys_non(std::move(proj_ptr_non));

    auto fft_ptr_lin{
        std::make_unique<muFFT::FFTWEngine>(resolutions)};
    auto proj_ptr_lin{std::make_unique<ProjectionFiniteStrainFast<Dim>>(
        std::move(fft_ptr_lin), lengths)};
    Cell sys_lin(std::move(proj_ptr_lin));

    using Mat_t_non = MaterialLinearOrthotropic<Dim>;
    using Mat_t_lin = MaterialLinearElastic1<Dim>;

    const Real Young{1e10}, Poisson{.3};
    const int con{2};
    const Real lambda{Young * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))};
    const Real mu{Young / (2 * (1 + Poisson))};

    auto & Material_soft_lin{
        Mat_t_lin::make(sys_lin, "soft_lin", Young, Poisson)};
    auto & Material_hard_lin{
        Mat_t_lin::make(sys_lin, "hard_lin", con * Young, Poisson)};

    std::vector<Real> input_soft;
    input_soft.push_back(lambda + 2 * mu);
    input_soft.push_back(lambda);
    input_soft.push_back(lambda + 2 * mu);
    input_soft.push_back(mu);

    std::vector<Real> input_hard;
    input_hard.push_back(con * (lambda + 2 * mu));
    input_hard.push_back(con * lambda);
    input_hard.push_back(con * (lambda + 2 * mu));
    input_hard.push_back(con * mu);

    auto & Material_soft{Mat_t_non::make(sys_non, "soft", input_soft)};
    auto & Material_hard{Mat_t_non::make(sys_non, "hard", input_hard)};

    for (const auto & pixel_id : sys_non.get_pixel_indices()) {
      if (pixel_id == 0) {
        Material_soft.add_pixel(pixel_id);
        Material_soft_lin.add_pixel(pixel_id);
      } else {
        Material_hard.add_pixel(pixel_id);
        Material_hard_lin.add_pixel(pixel_id);
      }
    }

    Grad_t<Dim> delF0{};
    delF0 << 1e-4, 5e-5, 5e-5, 0;  //, 0, 0, 0, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-5};
    const Uint maxiter{10};
    constexpr Verbosity verbose{Verbosity::Silent};

    LoadSteps_t grads;
    grads.push_back(delF0);
    KrylovSolverCG cg_lin(sys_lin, cg_tol, maxiter, verbose);
    Eigen::ArrayXXd res2{
        newton_cg(sys_lin, grads, cg_lin, newton_tol, equil_tol, verbose)[0]
            .grad};

    KrylovSolverCG cg_non{sys_non, cg_tol, maxiter, verbose};
    Eigen::ArrayXXd res1{
        newton_cg(sys_non, grads, cg_non, newton_tol, equil_tol, verbose)[0]
            .grad};

    Real error{abs(res1 - res2).mean()};
    BOOST_CHECK_LE(error, cg_tol);
  }

  BOOST_AUTO_TEST_CASE(orthotropic_threeD) {
    constexpr Dim_t Dim{threeD};

    const DynCcoord_t resolutions{5, 5, 5};
    const DynRcoord_t lengths{5, 5, 5};
    auto fft_ptr_non{
        std::make_unique<muFFT::FFTWEngine>(resolutions)};
    auto proj_ptr_non{std::make_unique<ProjectionFiniteStrainFast<Dim>>(
        std::move(fft_ptr_non), lengths)};
    Cell sys_non(std::move(proj_ptr_non));

    auto fft_ptr_lin{
        std::make_unique<muFFT::FFTWEngine>(resolutions)};
    auto proj_ptr_lin{std::make_unique<ProjectionFiniteStrainFast<Dim>>(
        std::move(fft_ptr_lin), lengths)};
    Cell sys_lin(std::move(proj_ptr_lin));

    using Mat_t_non = MaterialLinearOrthotropic<Dim>;
    using Mat_t_lin = MaterialLinearElastic1<Dim>;

    const Real Young{1e10}, Poisson{.3};
    const int con{2};
    const Real lambda{Young * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))};
    const Real mu{Young / (2 * (1 + Poisson))};

    auto & Material_soft_lin{
        Mat_t_lin::make(sys_lin, "soft_lin", Young, Poisson)};
    auto & Material_hard_lin{
        Mat_t_lin::make(sys_lin, "hard_lin", con * Young, Poisson)};

    std::vector<Real> input_soft;
    input_soft.push_back(lambda + 2 * mu);
    input_soft.push_back(lambda);
    input_soft.push_back(lambda);
    input_soft.push_back(lambda + 2 * mu);
    input_soft.push_back(lambda);
    input_soft.push_back(lambda + 2 * mu);
    input_soft.push_back(mu);
    input_soft.push_back(mu);
    input_soft.push_back(mu);

    std::vector<Real> input_hard;
    input_hard.push_back(con * (lambda + 2 * mu));
    input_hard.push_back(con * lambda);
    input_hard.push_back(con * lambda);
    input_hard.push_back(con * (lambda + 2 * mu));
    input_hard.push_back(con * lambda);
    input_hard.push_back(con * (lambda + 2 * mu));
    input_hard.push_back(con * mu);
    input_hard.push_back(con * mu);
    input_hard.push_back(con * mu);

    auto & Material_soft{Mat_t_non::make(sys_non, "soft", input_soft)};
    auto & Material_hard{Mat_t_non::make(sys_non, "hard", input_hard)};

    for (const auto & pixel_id : sys_non.get_pixel_indices()) {
      if (pixel_id == 0) {
        Material_soft.add_pixel(pixel_id);
        Material_soft_lin.add_pixel(pixel_id);
      } else {
        Material_hard.add_pixel(pixel_id);
        Material_hard_lin.add_pixel(pixel_id);
      }
    }

    Grad_t<Dim> delF0;
    delF0 << 1e-4, 5e-5, 5e-5, 0, 0, 0, 0, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-5};
    const Uint maxiter{10};
    constexpr Verbosity verbose{Verbosity::Silent};

    LoadSteps_t grads;
    grads.push_back(delF0);
    KrylovSolverCG cg_lin{sys_lin, cg_tol, maxiter, verbose};
    Eigen::ArrayXXd res2{
        newton_cg(sys_lin, delF0, cg_lin, newton_tol, equil_tol, verbose).grad};

    KrylovSolverCG cg_non{sys_non, cg_tol, maxiter, verbose};
    Eigen::ArrayXXd res1{
        newton_cg(sys_non, delF0, cg_non, newton_tol, equil_tol, verbose).grad};

    BOOST_CHECK_LE(abs(res1 - res2).mean(), cg_tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
