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
    : adaptable{adaptable}, w_adaptable{adaptable} {}

  /* ---------------------------------------------------------------------- */
  MatrixAdaptor::MatrixAdaptor(std::weak_ptr<MatrixAdaptable> adaptable)
      : w_adaptable{adaptable} {}

  /* ---------------------------------------------------------------------- */
  Index_t MatrixAdaptor::get_nb_dof() const {
    if (this->w_adaptable.expired()) {
      throw muGrid::RuntimeError{
          "This matrix adaptor does not belong to any matrix adaptable"};
    }
    return this->w_adaptable.lock()->get_nb_dof();
  }

  /* ---------------------------------------------------------------------- */
  void MatrixAdaptor::action_increment(EigenCVec_t delta_grad,
                                       const Real & alpha,
                                       EigenVec_t del_flux) const {
    if (this->w_adaptable.expired()) {
      throw muGrid::RuntimeError{
          "This matrix adaptor does not belong to any matrix adaptable"};
    }
    return this->w_adaptable.lock()->action_increment(delta_grad, alpha,
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

  /* ---------------------------------------------------------------------- */
  MatrixAdaptor MatrixAdaptable::get_weak_adaptor() {
    return MatrixAdaptor{
        std::weak_ptr<MatrixAdaptable>{this->shared_from_this()}};
  }

  /* ---------------------------------------------------------------------- */
  DenseEigenAdaptor::DenseEigenAdaptor(
      const Eigen::Ref<const Eigen::MatrixXd> matrix)
      : matrix{matrix} {
    if (this->matrix.rows() != this->matrix.cols()) {
      throw muGrid::RuntimeError(
          "Only square matrices can be used in adaptors");
    }
  }

  /* ---------------------------------------------------------------------- */
  DenseEigenAdaptor::DenseEigenAdaptor(const Index_t & nb_dof)
      : matrix{Eigen::MatrixXd::Zero(nb_dof, nb_dof)} {}

  /* ---------------------------------------------------------------------- */
  Index_t DenseEigenAdaptor::get_nb_dof() const { return this->matrix.rows(); }

  /* ---------------------------------------------------------------------- */
  void DenseEigenAdaptor::action_increment(EigenCVec_t delta_grad,
                                           const Real & alpha,
                                           EigenVec_t del_flux) {
    del_flux += alpha * this->matrix * delta_grad;
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::Communicator & DenseEigenAdaptor::get_communicator() const {
    return this->comm;
  }

  /* ---------------------------------------------------------------------- */
  Eigen::MatrixXd & DenseEigenAdaptor::get_matrix() { return this->matrix; }

  /* ---------------------------------------------------------------------- */
  const Eigen::MatrixXd & DenseEigenAdaptor::get_matrix() const {
    return this->matrix;
  }

}  // namespace muSpectre
