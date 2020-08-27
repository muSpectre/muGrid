/**
 * @file   matrix_adaptor.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   31 Jul 2020
 *
 * @brief  implementation for base-class for all sparse matrix representatitions
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

#include "matrix_adaptor.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  MatrixAdaptor::MatrixAdaptor(std::shared_ptr<MatrixAdaptable> adaptable)
      : adaptable{adaptable} {}

  /* ---------------------------------------------------------------------- */
  Index_t MatrixAdaptor::get_nb_dof() const {
    if (this->adaptable == nullptr) {
      throw muGrid::RuntimeError{
          "This matrix adaptor does not belong to any matrix adaptable"};
    }
    return this->adaptable->get_nb_dof();
  }

  /* ---------------------------------------------------------------------- */
  void MatrixAdaptor::stiffness_action_increment(EigenCVec_t delta_grad,
                                                 const Real & alpha,
                                                 EigenVec_t del_flux) const {
    if (this->adaptable == nullptr) {
      throw muGrid::RuntimeError{
          "This matrix adaptor does not belong to any matrix adaptable"};
    }
    return this->adaptable->stiffness_action_increment(delta_grad, alpha,
                                                       del_flux);
  }

  /* ---------------------------------------------------------------------- */
  Eigen::Index MatrixAdaptor::rows() const { return this->get_nb_dof(); }

  /* ---------------------------------------------------------------------- */
  Eigen::Index MatrixAdaptor::cols() const { return this->get_nb_dof(); }

  /* ---------------------------------------------------------------------- */
  MatrixAdaptor MatrixAdaptable::get_adaptor() {
    return MatrixAdaptor{this->shared_from_this()};
  }

}  // namespace muSpectre
