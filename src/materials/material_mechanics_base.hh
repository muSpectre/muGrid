/**
 * @file   material_mechanics_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   22 Jun 2020
 *
 * @brief  Mostly empty base class for all solid mechanics constitutive laws
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

#ifndef SRC_MATERIALS_MATERIAL_MECHANICS_BASE_HH_
#define SRC_MATERIALS_MATERIAL_MECHANICS_BASE_HH_

#include "common/muSpectre_common.hh"
#include "material_base.hh"

namespace muSpectre {

  class MaterialMechanicsBase : public MaterialBase {
   public:
    using Parent = MaterialBase;
    //! Default constructor
    MaterialMechanicsBase() = delete;

    // just use MaterialBase's constructor, this class is purely interface
    using Parent::Parent;

    //! Copy constructor
    MaterialMechanicsBase(const MaterialMechanicsBase & other) = delete;

    //! Move constructor
    MaterialMechanicsBase(MaterialMechanicsBase && other) = delete;

    //! Destructor
    virtual ~MaterialMechanicsBase() = default;

    //! Copy assignment operator
    MaterialMechanicsBase &
    operator=(const MaterialMechanicsBase & other) = delete;

    //! Move assignment operator
    MaterialMechanicsBase & operator=(MaterialMechanicsBase && other) = delete;

    virtual const Formulation & get_formulation() const = 0;
    virtual void set_formulation(const Formulation & form) = 0;

    muGrid::PhysicsDomain get_physics_domain() const final;

    /**
     * checks whether this material can be used in small strain formulation
     */
    void check_small_strain_capability() const;

    /**
     * returns the expected strain measure of the material
     */
    virtual StrainMeasure get_expected_strain_measure() const = 0;

    /**
     * returns a reference to the currently set solver type
     */
    const SolverType & get_solver_type() const;

    /**
     * set the solver type
     */
    void set_solver_type(const SolverType & solver_type);

   protected:
    SolverType solver_type{SolverType::Spectral};
  };

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_MECHANICS_BASE_HH_
