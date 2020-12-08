/**
 * @file   material_mechanics_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   28 Jul 2020
 *
 * @brief  implementation for MaterialMechanicsBase
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

#include "material_mechanics_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  muGrid::PhysicsDomain MaterialMechanicsBase::get_physics_domain() const {
    return muGrid::PhysicsDomain::mechanics();
  }

  /* ---------------------------------------------------------------------- */
  void MaterialMechanicsBase::check_small_strain_capability() const {
    if (not(is_objective(this->get_expected_strain_measure()))) {
      std::stringstream err_str{};
      err_str
          << "The material expected strain measure is: "
          << this->get_expected_strain_measure()
          << ", while in small strain the required strain measure should be "
             "objective (in order to be obtainable from infinitesimal strain)."
          << " Accordingly, this material is not meant to be utilized in "
             "small strain formulation"
          << std::endl;
      throw(muGrid::RuntimeError(err_str.str()));
    }
  }

  /* ---------------------------------------------------------------------- */
  const SolverType & MaterialMechanicsBase::get_solver_type() const {
    return this->solver_type;
  }

  /* ---------------------------------------------------------------------- */
  void MaterialMechanicsBase::set_solver_type(const SolverType & solver_type) {
    this->solver_type = solver_type;
  }
}  // namespace muSpectre
