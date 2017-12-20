/**
 * file   solver_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2017
 *
 * @brief  definitions for solvers
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

#include <iostream>
#include <memory>

#include "solver/solver_base.hh"
#include "common/field.hh"
#include "common/iterators.hh"


namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys, const GradIncrements<DimM> & delFs,
            const Real cg_tol, const Real newton_tol, Uint maxiter,
            bool verbose) {
    auto solver_fields{std::make_unique<GlobalFieldCollection<DimS, DimM>>()};
    solver_fields->initialise(sys.get_resolutions());

    if (maxiter == 0) {
      maxiter = sys.size()*DimM*DimM*10;
    }
    if (verbose) {
      //setup of algorithm 5.2 in Nocedal, Numerical Optimization (p. 111)
      std::cout << "Algo 5.2 with newton_tol = " << newton_tol << ", cg_tol = "
                << cg_tol << " maxiter = " << maxiter << " and ΔF =" <<std::endl;
      for (auto&& tup: akantu::enumerate(delFs)) {
        auto && counter{std::get<0>(tup)};
        auto && grad{std::get<1>(tup)};
        std::cout << "Step " << counter + 1 << ":" << std::endl
                  << grad << std::endl;
      }
    }

    // initialise F = I
    auto & F{sys.get_strain()};
    F.get_map() = Matrices::I2<DimM>();

    // initialise materials
    constexpr bool need_tangent{true};
    sys.initialise_materials(need_tangent);

    // Corresponds to symbol δF or δε
    auto & incrF{make_field
        <typename MaterialBase<DimS, DimM>::StrainField_t>("δF",
                                                           *solver_fields)};

    // field to store the rhs for cg calculations
    auto & rhs{make_field
        <typename MaterialBase<DimS, DimM>::StrainField_t>("rhs",
                                                           *solver_fields)};
    Grad_t<DimM> previous_grad{Grad_t<DimM>::Zero()};
    for (const auto & delF: delFs) { //incremental loop
      // apply macroscopic strain increment
      for (auto && grad: F.get_map()) {
        grad += delF - previous_grad;
      }

      // optain material response
      auto res_tup{sys.evaluate_stress_tangent(F)};
      auto & P{std::get<0>(res_tup)};
      auto & K{std::get<1>(res_tup)};

      for (Uint i = 0; i < maxiter; ++i) {
        
      }


      // update previous gradient
      previous_grad = delF;
    }


  }

  // template typename SystemBase<twoD, twoD>::StrainField_t &
  // newton_cg (SystemBase<twoD, twoD> & sys, const GradIncrements<twoD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            bool verbose);

  // template typename SystemBase<twoD, threeD>::StrainField_t &
  // newton_cg (SystemBase<twoD, threeD> & sys, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            bool verbose);

  // template typename SystemBase<threeD, threeD>::StrainField_t &
  // newton_cg (SystemBase<threeD, threeD> & sys, const GradIncrements<threeD>& delF0,
  //            const Real cg_tol, const Real newton_tol, Uint maxiter,
  //            bool verbose);

}  // muSpectre
