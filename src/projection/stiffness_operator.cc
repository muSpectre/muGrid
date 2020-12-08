/**
 * @file   stiffness_operator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   20 Jul 2020
 *
 * @brief  implementation for stiffness operator member functions
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

#include "stiffness_operator.hh"

#include <libmugrid/field_collection_global.hh>
#include <libmugrid/exception.hh>
#include <libmugrid/iterators.hh>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  StiffnessOperator::StiffnessOperator(
      const Index_t & displacement_rank,
      std::shared_ptr<muGrid::GradientOperatorBase> gradient_operator,
      const std::vector<Real> & quadrature_weights,
      const Formulation & formulation)
      : displacement_rank{displacement_rank},
        gradient_operator{gradient_operator},
        nb_displacement_components{
            muGrid::ipow(this->gradient_operator->get_spatial_dim(),
                         this->displacement_rank)},
        quadrature_weights{quadrature_weights}, formulation{formulation} {
    if (static_cast<Index_t>(this->quadrature_weights.size()) !=
        gradient_operator->get_nb_pixel_quad_pts()) {
      std::stringstream error_message{};
      error_message << "You provided " << this->quadrature_weights.size()
                    << " weights, but the gradient operator has "
                    << gradient_operator->get_nb_pixel_quad_pts()
                    << " quadrature points per pixel.";
      throw muGrid::RuntimeError{error_message.str()};
    }
    // TODO(junge, ladecky): check that nb_dof is a multiple of the
    // gradient's, NOTE: I don't get, what should be tested {ladecky}
    //
    // nb_component_per_pixel
  }

  void StiffnessOperator::prepare_application(
      const muGrid::TypedFieldBase<Real> & displacement,
      const muGrid::TypedFieldBase<Real> & force) {
    const Index_t spatial_dim{this->gradient_operator->get_spatial_dim()};
    const Index_t nb_nodal_dof{
        muGrid::ipow(spatial_dim, this->displacement_rank)};
    const Index_t nb_quad_dof{
        muGrid::ipow(spatial_dim, 1 + this->displacement_rank)};

    if (nb_nodal_dof != displacement.get_nb_components()) {
      std::stringstream error_message{};
      error_message << "You provided " << nb_nodal_dof
                    << " number of DOF per node, but the displacement has "
                    << displacement.get_nb_components()
                    << " number of DOF per sub point.";
      throw muGrid::RuntimeError{error_message.str()};
    }
    if (nb_nodal_dof != force.get_nb_components()) {
      std::stringstream error_message{};
      error_message << "You provided " << nb_nodal_dof
                    << " number of DOF per node, but the force has "
                    << force.get_nb_components()
                    << " number of DOF per sub point.";
      throw muGrid::RuntimeError{error_message.str()};
    }

    if (this->quad_field == nullptr) {
      this->quad_field = std::make_shared<
          muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>>(
          displacement.get_collection().generate_unique_name(), nb_quad_dof, 1,
          muGrid::IterUnit::SubPt, displacement.get_collection(), QuadPtTag);
    }

    // apply gradiant
    this->gradient_operator->apply_gradient(displacement,
                                            this->quad_field->get_field());
  }

  /* ---------------------------------------------------------------------- */
  void StiffnessOperator::apply(
      const Eigen::Ref<const Eigen::MatrixXd> & material_properties,
      const muGrid::TypedFieldBase<Real> & displacement,
      muGrid::TypedFieldBase<Real> & force) {
    switch (this->formulation) {
    case Formulation::small_strain: {
      this->apply_worker<Formulation::small_strain>(material_properties,
                                                    displacement, force);
      break;
    }
    case Formulation::finite_strain: {
      // fall-through
    }
    case Formulation::not_set: {
      this->apply_worker<Formulation::not_set>(material_properties,
                                               displacement, force);
      break;
    }
    default:
      throw StiffnessError{"Can't handle the formulation you've chosen"};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Formulation Form>
  void StiffnessOperator::apply_worker(
      const Eigen::Ref<const Eigen::MatrixXd> & material_properties,
      const muGrid::TypedFieldBase<Real> & displacement,
      muGrid::TypedFieldBase<Real> & force) {
    this->prepare_application(displacement, force);

    const Index_t spatial_dim{this->gradient_operator->get_spatial_dim()};
    const Index_t nb_quad_dof{
        muGrid::ipow(spatial_dim, 1 + this->displacement_rank)};

    if (nb_quad_dof != material_properties.rows() or
        nb_quad_dof != material_properties.cols()) {
      std::stringstream error_message{};
      error_message << "You provided " << nb_quad_dof
                    << " number of gradient DOF per quadrature point,"
                       " but the material_properties has"
                    << material_properties.rows() << "rows and "
                    << material_properties.cols() << "columns ";
      throw muGrid::RuntimeError{error_message.str()};
    }

    const Index_t nb_quad_pts{this->gradient_operator->get_nb_pixel_quad_pts()};

    for (auto && index_grad_val :
         this->quad_field->get_map().enumerate_indices()) {
      auto && id{std::get<0>(index_grad_val)};
      auto && grad_val{std::get<1>(index_grad_val)};

      grad_val = (material_properties *
                  internal::GradientTransformer<Form>::transform(grad_val) *
                  this->quadrature_weights[id % nb_quad_pts])
                     .eval();
    }

    // apply transpose gradient
    this->gradient_operator->apply_transpose(this->quad_field->get_field(),
                                             force);
  }

  /* ---------------------------------------------------------------------- */
  void StiffnessOperator::apply(
      const muGrid::TypedFieldBase<Real> & material_properties,
      const muGrid::TypedFieldBase<Real> & displacement,
      muGrid::TypedFieldBase<Real> & force) {
    force.set_zero();
    this->apply_increment(material_properties, displacement, 1., force);
  }

  /* ---------------------------------------------------------------------- */
  void StiffnessOperator::apply_increment(
      const muGrid::TypedFieldBase<Real> & material_properties,
      const muGrid::TypedFieldBase<Real> & displacement, const Real & alpha,
      muGrid::TypedFieldBase<Real> & force) {
    switch (this->formulation) {
    case Formulation::small_strain: {
      this->apply_increment_worker<Formulation::small_strain>(
          material_properties, displacement, alpha, force);
      break;
    }
    case Formulation::finite_strain: {
      // fall-through
    }
    case Formulation::not_set: {
      this->apply_increment_worker<Formulation::not_set>(
          material_properties, displacement, alpha, force);
      break;
    }
    default:
      throw StiffnessError{"Can't handle the formulation you've chosen"};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Formulation Form>
  void StiffnessOperator::apply_increment_worker(
      const muGrid::TypedFieldBase<Real> & material_properties,
      const muGrid::TypedFieldBase<Real> & displacement, const Real & alpha,
      muGrid::TypedFieldBase<Real> & force) {
    this->prepare_application(displacement, force);

    const Index_t spatial_dim{this->gradient_operator->get_spatial_dim()};
    const Index_t nb_quad_dof{
        muGrid::ipow(spatial_dim, 1 + this->displacement_rank)};

    if (nb_quad_dof * nb_quad_dof != material_properties.get_nb_components()) {
      std::stringstream error_message{};
      error_message
          << " Expected material_properties size per quadrature point is,"
          << nb_quad_dof * nb_quad_dof << " but the material_properties has"
          << material_properties.get_nb_components() << "number of components ";
      throw muGrid::RuntimeError{error_message.str()};
    }

    muGrid::FieldMap<Real, Mapping::Const> tangent_moduli{
        material_properties, nb_quad_dof, IterUnit::SubPt};

    for (auto && tangent_index_grad_val :
         akantu::zip(tangent_moduli, this->quad_field->get_map())) {
      auto && tangent{std::get<0>(tangent_index_grad_val)};
      auto && grad_val{std::get<1>(tangent_index_grad_val)};

      grad_val =
          (tangent * internal::GradientTransformer<Form>::transform(grad_val))
              .eval();
    }

    // apply transpose gradient
    this->gradient_operator->apply_transpose_increment(
        this->quad_field->get_field(), alpha, force, this->quadrature_weights);
  }

  /* ---------------------------------------------------------------------- */
  void StiffnessOperator::apply_increment(
      const muGrid::TypedFieldBase<Real> & material_properties,
      EigenCVec_t displacement, const Real & alpha, EigenVec_t force) {
    auto delta_displacement_field_ptr{muGrid::WrappedField<Real>::make_const(
        "delta_disp", material_properties.get_collection(),
        this->nb_displacement_components, displacement, NodalPtTag)};
    muGrid::WrappedField<Real> delta_force_field{
        "delta_force", material_properties.get_collection(),
        this->nb_displacement_components, force, NodalPtTag};
    this->apply_increment(material_properties, *delta_displacement_field_ptr,
                          alpha, delta_force_field);
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<muGrid::GradientOperatorBase>
  StiffnessOperator::get_gradient_operator() {
    return this->gradient_operator;
  }

  /* ---------------------------------------------------------------------- */
  void StiffnessOperator::apply_divergence(
      const muGrid::TypedFieldBase<Real> & quadrature_point_field,
      muGrid::TypedFieldBase<Real> & nodal_field) const {
    this->gradient_operator->apply_transpose(
        quadrature_point_field, nodal_field, this->quadrature_weights);
  }

}  // namespace muSpectre
