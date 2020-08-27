/**
 * @file   solver_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   26 Jun 2020
 *
 * @brief  Implementation for SolverBase methods
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

#include "solver_base.hh"
#include "materials/material_mechanics_base.hh"

#include <type_traits>
#include <memory>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverBase::SolverBase(std::shared_ptr<CellData> cell_data,
                         const muGrid::Verbosity & verbosity)
      : Parent{}, cell_data{cell_data}, verbosity{verbosity} {};

  /* ---------------------------------------------------------------------- */
  void SolverBase::set_formulation(const Formulation & formulation) {
    this->formulation = formulation;
  }

  /* ---------------------------------------------------------------------- */
  const Formulation & SolverBase::get_formulation() const {
    return this->formulation;
  }

  /* ---------------------------------------------------------------------- */
  void SolverBase::reset_counter() { this->counter = 0; }

  /* ---------------------------------------------------------------------- */
  const Int & SolverBase::get_counter() const { return this->counter; }

  void check_material_formulation(const CellData::Material_ptr mat,
                                  const Formulation & form,
                                  const muGrid::PhysicsDomain & domain) {
    if (domain == muGrid::PhysicsDomain::mechanics()) {
      try {
        std::dynamic_pointer_cast<MaterialMechanicsBase>(mat);
      } catch (const std::bad_cast & error) {
        std::stringstream error_stream{};
        error_stream << "Material '" << mat->get_name()
                     << "' cannot be used as a mechanics material.";
        throw SolverError{error_stream.str()};
      }
      if (std::dynamic_pointer_cast<MaterialMechanicsBase>(mat)
              ->get_formulation() != form) {
        std::stringstream error_stream{};
        error_stream << "The material '" << mat->get_name()
                     << "', has formulation "
                     << std::dynamic_pointer_cast<MaterialMechanicsBase>(mat)
                            ->get_formulation()
                     << ", but " << form << " is required.";
        throw MaterialError{error_stream.str()};
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  auto SolverBase::evaluate_stress(const muGrid::PhysicsDomain & domain)
      -> const MappedField_t & {
    if (not this->is_initialised) {
      this->initialise_cell();
    }
    for (auto && mat : this->cell_data->get_domain_materials().at(domain)) {
      check_material_formulation(mat, this->get_formulation(), domain);
      mat->compute_stresses(this->grads.at(domain)->get_field(),
                            this->fluxes.at(domain)->get_field());
    }
    return *this->fluxes.at(domain);
  }

  /* ---------------------------------------------------------------------- */
  auto SolverBase::evaluate_stress_tangent(const muGrid::PhysicsDomain & domain)
      -> std::tuple<const MappedField_t &, const MappedField_t &> {
    if (not this->is_initialised) {
      this->initialise_cell();
    }
    for (auto && mat : this->cell_data->get_domain_materials().at(domain)) {
      check_material_formulation(mat, this->get_formulation(), domain);
      mat->compute_stresses_tangent(this->grads.at(domain)->get_field(),
                                    this->fluxes.at(domain)->get_field(),
                                    this->tangents.at(domain)->get_field());
    }
    return std::tie(*this->fluxes.at(domain), *this->tangents.at(domain));
  }

  /* ---------------------------------------------------------------------- */
  Int & SolverBase::get_counter() { return this->counter; }

  /* ---------------------------------------------------------------------- */
  const muFFT::Communicator & SolverBase::get_communicator() const {
    return this->cell_data->get_communicator();
  }

}  // namespace muSpectre
