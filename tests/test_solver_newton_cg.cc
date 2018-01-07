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
#include "system/system_factory.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(newton_cg_tests);

  BOOST_AUTO_TEST_CASE(manual_construction_test) {
    // constexpr Dim_t dim{twoD};
    constexpr Dim_t dim{threeD};

    // constexpr Ccoord_t<dim> resolutions{3, 3};
    // constexpr Rcoord_t<dim> lengths{2.3, 2.7};
    constexpr Ccoord_t<dim> resolutions{5, 5, 5};
    constexpr Rcoord_t<dim> lengths{5, 5, 5};
    auto fft_ptr{std::make_unique<FFTW_Engine<dim, dim>>(resolutions, lengths)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<dim, dim>>(std::move(fft_ptr))};
    SystemBase<dim, dim> sys(std::move(proj_ptr));

    using Mat_t = MaterialHyperElastic1<dim, dim>;
    //const Real Young{210e9}, Poisson{.33};
    const Real Young{1.0030648180242636}, Poisson{0.29930675909878679};
    // const Real lambda{Young*Poisson/((1+Poisson)*(1-2*Poisson))};
    // const Real mu{Young/(2*(1+Poisson))};
    auto Material_hard = std::make_unique<Mat_t>("hard", 10*Young, Poisson);
    auto Material_soft = std::make_unique<Mat_t>("soft", Young, Poisson);

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
    sys.initialise();

    Grad_t<dim> delF0;
    delF0 << 0, 1., 0, 0, 0, 0, 0, 0, 0;
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5};
    constexpr Uint maxiter{CcoordOps::get_size(resolutions)*ipow(dim, secondOrder)*10};
    constexpr bool verbose{false};
    constexpr Formulation form{Formulation::finite_strain};

    GradIncrements<dim> grads; grads.push_back(delF0);
    Eigen::ArrayXXd res1{de_geus(sys, grads, form, cg_tol, newton_tol, maxiter, verbose).eigen()};

    Eigen::ArrayXXd res2{newton_cg(sys, grads, form, cg_tol, newton_tol, maxiter, verbose).eigen()};
    BOOST_CHECK_LE(abs(res1-res2).mean(), cg_tol);
  }

  BOOST_AUTO_TEST_CASE(small_strain_patch_test) {
    constexpr Dim_t dim{twoD};
    using Ccoord = Ccoord_t<dim>;
    using Rcoord = Rcoord_t<dim>;
    constexpr Ccoord resolutions{3, 3};
    constexpr Rcoord lengths{3, 3};
    constexpr Formulation form{Formulation::small_strain};

    // number of layers in the hard material
    constexpr Uint nb_lays{1};
    constexpr Real contrast{2};
    static_assert(nb_lays < resolutions[0],
                  "the number or layers in the hard material must be smaller "
                  "than the total number of layers in dimension 0");

    auto sys{make_system(resolutions, lengths, form)};

    using Mat_t = MaterialHyperElastic1<dim, dim>;
    constexpr Real Young{2}, Poisson{.33};
    auto material_hard{std::make_unique<Mat_t>("hard", contrast*Young, Poisson)};
    auto material_soft{std::make_unique<Mat_t>("soft",          Young, Poisson)};

    for (const auto & pixel: sys) {
      if (pixel[0] < Dim_t(nb_lays)) {
        material_hard->add_pixel(pixel);
      } else {
        material_soft->add_pixel(pixel);
      }
    }

    sys.add_material(std::move(material_hard));
    sys.add_material(std::move(material_soft));
    sys.initialise();

    Grad_t<dim> delEps0;
    constexpr Real eps0 = 1.;
    delEps0 << eps0, 0, 0, 0;

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5};
    constexpr Uint maxiter{CcoordOps::get_size(resolutions)*ipow(dim, secondOrder)*10};
    constexpr bool verbose{false};

    auto & result = newton_cg(sys, delEps0, form, cg_tol, newton_tol, maxiter, verbose);
    if (verbose) {
      std::cout << "result:" << std::endl << result.eigen() << std::endl;
      std::cout << "mean strain = " << std::endl << result.get_map().mean() << std::endl;
    }

    /**
     *  verification of resultant strains: subscript ₕ for hard and ₛ
     *  for soft, Nₕ is nb_lays and Nₜₒₜ is resolutions, k is contrast
     *
     *     Δl = εl = Δlₕ + Δlₛ = εₕlₕ+εₛlₛ
     *  => ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ
     *
     *  σ is constant across all layers
     *        σₕ = σₛ
     *  => Eₕ εₕ = Eₛ εₛ
     *  => εₕ = 1/k εₛ
     *  => ε / (1/k Nₕ/Nₜₒₜ + (Nₜₒₜ-Nₕ)/Nₜₒₜ) = εₛ
     */
    constexpr Real factor{1/contrast * Real(nb_lays)/resolutions[0]
        + 1.-nb_lays/Real(resolutions[0])};
    constexpr Real eps_soft{eps0/factor};
    constexpr Real eps_hard{eps_soft/contrast};
    if (verbose) {
      std::cout << "εₕ = " << eps_hard << ", εₛ = " << eps_soft << std::endl;
      std::cout << "ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ" << std::endl;
    }
    Grad_t<dim> Eps_hard; Eps_hard << eps_hard, 0, 0, 0;
    Grad_t<dim> Eps_soft; Eps_soft << eps_soft, 0, 0, 0;

    for (const auto & pixel: sys) {
      if (pixel[0] < Dim_t(nb_lays)) {
        BOOST_CHECK_LE((Eps_hard-result.get_map()[pixel]).norm(), tol);
      } else {
        BOOST_CHECK_LE((Eps_soft-result.get_map()[pixel]).norm(), tol);
      }
    }


  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
