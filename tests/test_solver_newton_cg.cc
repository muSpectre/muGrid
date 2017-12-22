/**
 * file   test_solver_newton_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Tests for the standard Newton-Raphson + Conjugate Gradient solver
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

#include "tests.hh"
#include "solver/solvers.hh"
#include "fft/fftw_engine.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "materials/material_hyper_elastic1.hh"
#include "common/iterators.hh"
#include "common/ccoord_operations.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(newton_cg_tests);

  BOOST_AUTO_TEST_CASE(manual_construction_test) {
    constexpr Dim_t dim{twoD};

    constexpr Ccoord_t<dim> resolutions{3, 3};
    constexpr Rcoord_t<dim> lengths{2.3, 2.7};
    auto fft_ptr{std::make_unique<FFTW_Engine<dim, dim>>(resolutions, lengths)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(std::move(fft_ptr))};
    SystemBase<dim, dim> sys(std::move(proj_ptr));

    using Mat_t = MaterialHyperElastic1<dim, dim>;
    const Real Young{210e9}, Poisson{.33};
    // const Real lambda{Young*Poisson/((1+Poisson)*(1-2*Poisson))};
    // const Real mu{Young/(2*(1+Poisson))};
    auto Material_hard = std::make_unique<Mat_t>("hard", Young, Poisson);
    auto Material_soft = std::make_unique<Mat_t>("soft", Young*.1, Poisson);

    for (auto && tup: akantu::enumerate(sys)) {
      auto && pixel = std::get<1>(tup);
      if (std::get<0>(tup) == 0) {
        Material_hard->add_pixel(pixel);
      } else {
        Material_soft->add_pixel(pixel);
      }
    }
    sys.add_material(std::move(Material_hard));
    sys.add_material(std::move(Material_soft));

    Grad_t<dim> delF0;
    delF0 << 0, .1, 0, 0;
    constexpr Real cg_tol{1e-5}, newton_tol{1e-5};
    constexpr Uint maxiter
      {CcoordOps::get_size(resolutions)*ipow(dim, secondOrder)*10};
    constexpr bool verbose{true};

    GradIncrements<dim> grads; grads.push_back(delF0);
    de_geus(sys, grads, cg_tol, newton_tol, maxiter, verbose);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
