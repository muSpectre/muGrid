/**
 * @file   discretisation.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   03 Aug 2020
 *
 * @brief  Implementation of discretisation object. Can produce Stiffnes
 * operators
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "discretisation.hh"

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  Discretisation::Discretisation(const std::shared_ptr<FEMStencilBase> stencil)
      : cell_ptr(stencil->get_cell_ptr()), stencil_ptr(stencil) {}
  /* ---------------------------------------------------------------------- */
  StiffnessOperator Discretisation::get_stiffness_operator(
      const Index_t displacement_rank) const {
    auto grad{this->stencil_ptr->get_gradient_operator()};
    StiffnessOperator K_operator(displacement_rank, grad,
                                 this->stencil_ptr->get_quadrature_weights());

    return K_operator;
  }

  /* ---------------------------------------------------------------------- */
  Index_t Discretisation::get_nb_quad_pts() const {
    return this->stencil_ptr->get_gradient_operator()->get_nb_pixel_quad_pts();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Discretisation::get_nb_nodal_pts() const {
    return this->stencil_ptr->get_gradient_operator()->get_nb_pixel_nodal_pts();
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<CellData> Discretisation::get_cell() {
    return this->cell_ptr;
  }
  /* ---------------------------------------------------------------------- */
  std::unique_ptr<muGrid::RealField, muGrid::FieldDestructor<muGrid::Field>>
  Discretisation::compute_impulse_response(
      const Index_t & displacement_rank,
      Eigen::Ref<const Eigen::MatrixXd> ref_material_properties) const {
    auto & dim{this->cell_ptr->get_spatial_dim()};
    const Index_t nb_nodal_pts{
        this->stencil_ptr->get_gradient_operator()->get_nb_pixel_nodal_pts()};

    const Index_t nb_impulse_component_per_pixel{
        nb_nodal_pts * muGrid::ipow(dim, displacement_rank)};

    using MappedField_t =
        muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>;

    auto & collection{this->cell_ptr->get_fields()};

    // temporary field to hold the response to an impulse on a given dof
    const std::string dof_response_name{"dof_response"};
    MappedField_t dof_response(dof_response_name,
                               nb_impulse_component_per_pixel, 1,
                               IterUnit::Pixel, collection, PixelTag);
    // temporary field to hold an impulse on a given dof
    const std::string dof_impulse_name{"dof_impulse"};
    MappedField_t dof_impulse{dof_impulse_name,
                              nb_impulse_component_per_pixel,
                              1,
                              IterUnit::Pixel,
                              collection,
                              PixelTag};
    const std::string impulse_response_name{"impulse_response"};
    MappedField_t impulse_response{impulse_response_name,
                                   nb_impulse_component_per_pixel,
                                   nb_impulse_component_per_pixel,
                                   IterUnit::Pixel,
                                   collection,
                                   PixelTag};

    auto && stiffness_operator{this->get_stiffness_operator(displacement_rank)};
    for (Index_t dof_id{0}; dof_id < nb_impulse_component_per_pixel; ++dof_id) {
      dof_impulse.get_map()[0] =
          Eigen::MatrixXd::Identity(nb_impulse_component_per_pixel,
                                    nb_impulse_component_per_pixel)
              .col(dof_id);
      stiffness_operator.apply(ref_material_properties, dof_impulse.get_field(),
                               dof_response.get_field());
      for (auto && dof_full :
           akantu::zip(dof_response.get_map(), impulse_response.get_map())) {
        auto && dof{std::get<0>(dof_full)};
        auto && full{std::get<1>(dof_full).col(dof_id)};
        full = dof;
      }
    }

    collection.pop_field(dof_response_name);
    collection.pop_field(dof_impulse_name);
    auto field_ptr{collection.pop_field(impulse_response_name)};
    auto * raw_ptr{static_cast<muGrid::RealField *>(field_ptr.get())};

    std::unique_ptr<muGrid::RealField, muGrid::FieldDestructor<muGrid::Field>>
        return_ptr{raw_ptr, std::move(field_ptr.get_deleter())};
    field_ptr.release();
    return return_ptr;
  }

}  // namespace muSpectre
