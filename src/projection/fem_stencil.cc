/**
 * @file   fem_stencil.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   02 Aug 2020
 *
 * @brief  Implementation of member functions for the fem stencill
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

#include "fem_stencil.hh"

#include <libmugrid/exception.hh>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  FEMStencilBase::FEMStencilBase(const std::vector<Real> & quadrature_weights,
                                 std::shared_ptr<CellData> cell)
      : quadrature_weights{quadrature_weights}, cell{cell} {}

  /* ---------------------------------------------------------------------- */
  const std::vector<Real> & FEMStencilBase::get_quadrature_weights() const {
    return this->quadrature_weights;
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<CellData> FEMStencilBase::get_cell_ptr() const {
    return this->cell;
  }

  /* ---------------------------------------------------------------------- */
  template <class GradientOperator>
  FEMStencil<GradientOperator>::FEMStencil(
      const Index_t & nb_quad_pts_per_element, const Index_t & nb_elements,
      const Index_t & nb_element_nodal_pts, const Index_t & nb_pixel_nodal_pts,
      const std::vector<std::vector<Eigen::MatrixXd>> & shape_fn_gradients,
      const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> &
          nodal_pts,
      const std::vector<Real> & quadrature_weights,
      std::shared_ptr<CellData> cell)
      : Parent{quadrature_weights, cell},
        gradient_operator{std::make_shared<GradientOperator>(
            cell->get_spatial_dim(), nb_quad_pts_per_element, nb_elements,
            nb_element_nodal_pts, nb_pixel_nodal_pts, shape_fn_gradients,
            nodal_pts)} {
    this->cell->set_nb_quad_pts(
        this->gradient_operator->get_nb_pixel_quad_pts());
    this->cell->set_nb_nodal_pts(
        this->gradient_operator->get_nb_pixel_nodal_pts());
  }

  /* ---------------------------------------------------------------------- */
  template <class GradientOperator>
  std::shared_ptr<muGrid::GradientOperatorBase>
  FEMStencil<GradientOperator>::get_gradient_operator() {
    return this->gradient_operator;
  }

  /* ---------------------------------------------------------------------- */
  template <class GradientOperator>
  const std::shared_ptr<muGrid::GradientOperatorBase>
  FEMStencil<GradientOperator>::get_gradient_operator() const {
    return this->gradient_operator;
  }

  /* ---------------------------------------------------------------------- */
  template <class GradientOperator>
  Index_t FEMStencil<GradientOperator>::get_nb_pixel_quad_pts() const {
    return this->gradient_operator->get_nb_pixel_quad_pts();
  }

  /* ---------------------------------------------------------------------- */
  template <class GradientOperator>
  Index_t FEMStencil<GradientOperator>::get_nb_pixel_nodal_pts() const {
    return this->gradient_operator->get_nb_pixel_quad_pts();
  }

  /* ---------------------------------------------------------------------- */
  template class FEMStencil<muGrid::GradientOperatorDefault>;

}  // namespace muSpectre
