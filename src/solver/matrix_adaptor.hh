/**
 * @file   matrix_adaptor.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jul 2020
 *
 * @brief  base class providing a sparse matrix interface to homogenisation
 *         problem representations
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

#ifndef SRC_SOLVER_MATRIX_ADAPTOR_HH_
#define SRC_SOLVER_MATRIX_ADAPTOR_HH_

#include "common/muSpectre_common.hh"

#include <libmugrid/exception.hh>
#include <libmugrid/communicator.hh>

#include <Eigen/IterativeLinearSolvers>

#include <memory>
namespace muSpectre {
  // forward declarations
  class MatrixAdaptor;
  class MatrixAdaptable;

}  // namespace muSpectre

namespace Eigen {
  namespace internal {
    using Real = muSpectre::Real;  //!< universal real value type
    template <>
    struct traits<muSpectre::MatrixAdaptor>
        : public Eigen::internal::traits<Eigen::SparseMatrix<Real>> {};
  }  // namespace internal

}  // namespace Eigen

namespace muSpectre {

  class MatrixAdaptor : public Eigen::EigenBase<MatrixAdaptor> {
   protected:
    friend MatrixAdaptable;
    //! Constructor
    explicit MatrixAdaptor(std::shared_ptr<MatrixAdaptable> adaptable);
    explicit MatrixAdaptor(std::weak_ptr<MatrixAdaptable> adaptable);

    //! Copy constructor
    MatrixAdaptor(const MatrixAdaptor & other) = default;

   public:
    //! Default constructor
    MatrixAdaptor() = default;

    //! Move constructor
    MatrixAdaptor(MatrixAdaptor && other) = default;

    //! Copy assignment operator
    MatrixAdaptor & operator=(const MatrixAdaptor & other) = default;

    //! Move assignment operator
    MatrixAdaptor & operator=(MatrixAdaptor && other) = default;

    //! Destructor
    virtual ~MatrixAdaptor() = default;

    using Scalar = Real;           //!< sparse matrix traits
    using RealScalar = Real;       //!< sparse matrix traits
    using StorageIndex = Index_t;  //!< sparse matrix traits
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      RowsAtCompileTime = Eigen::Dynamic,
      MaxRowsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
    };

    //! Ref to input/output vector
    using EigenVec_t = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! Ref to input vector
    using EigenCVec_t =
        Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    Index_t get_nb_dof() const;

    //! returns the number of logical rows
    Eigen::Index rows() const;
    //! returns the number of logical columns
    Eigen::Index cols() const;

    /**
     * evaluates the directional stiffness action contribution and increments
     * the flux field (this corresponds to G:K:δF (note the negative sign in
     * de Geus 2017, http://dx.doi.org/10.1016/j.cma.2016.12.032). and then
     * adds it do the values already in del_flux, scaled by alpha (i.e.,
     * del_flux += alpha*Q:K:δgrad. This function should not be used directly,
     * as it does absolutely no input checking. Rather, it is meant to be
     * called by the scaleAndAddTo function in the in Eigen solvers
     */
    void action_increment(EigenCVec_t delta_grad, const Real & alpha,
                          EigenVec_t del_flux) const;

    //! implementation of the evaluation
    template <typename Rhs>
    Eigen::Product<MatrixAdaptor, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> & x) const {
      return Eigen::Product<MatrixAdaptor, Rhs, Eigen::AliasFreeProduct>(
          *this, x.derived());
    }
    std::shared_ptr<MatrixAdaptable> adaptable{nullptr};
    std::weak_ptr<MatrixAdaptable> w_adaptable{};
  };

  class MatrixAdaptable : public std::enable_shared_from_this<MatrixAdaptable> {
   public:
    //! Default constructor
    MatrixAdaptable() = default;

    //! Copy constructor
    MatrixAdaptable(const MatrixAdaptable & other) = default;

    //! Move constructor
    MatrixAdaptable(MatrixAdaptable && other) = default;

    //! Copy assignment operator
    MatrixAdaptable & operator=(const MatrixAdaptable & other) = delete;

    //! Move assignment operator
    MatrixAdaptable & operator=(MatrixAdaptable && other) = delete;

    //! Destructor
    virtual ~MatrixAdaptable() = default;

    //! Ref to input/output vector
    using EigenVecRef = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! Ref to input vector
    using EigenCVecRef =
        Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    virtual Index_t get_nb_dof() const = 0;

    /**
     * evaluates the directional stiffness action contribution and increments
     * the flux field (this corresponds to G:K:δF (note the negative sign in de
     * Geus 2017, http://dx.doi.org/10.1016/j.cma.2016.12.032). and then adds it
     * do the values already in del_flux, scaled by alpha (i.e., del_flux +=
     * alpha*Q:K:δgrad. This function should not be used directly, as it does
     * absolutely no input checking. Rather, it is meant to be called by the
     * scaleAndAddTo function in the in Eigen solvers
     */
    virtual void action_increment(EigenCVecRef delta_grad, const Real & alpha,
                                  EigenVecRef del_flux) = 0;
    //! return the communicator object
    virtual const muGrid::Communicator & get_communicator() const = 0;

    /**
     * create a matrix adaptor satisfying the eigen matrix interface. The
     * returned Adaptor keeps this adaptable alive. This can cause shared_ptr
     * cycles.
     */
    MatrixAdaptor get_adaptor();

    /**
     * create a matrix adaptor satisfying the eigen matrix interface. The
     * returned Adaptor does not keep this adaptable alive. This avoids
     * shared_ptr cycles, but requires you to guarantee sufficient lifetime of
     * the Adaptable.
     */
    MatrixAdaptor get_weak_adaptor();
  };

  /**
   * matrix-adaptor for dense eigen matrix. For debugging and simple testing
   */
  class DenseEigenAdaptor : public MatrixAdaptable {
   public:
    using Parent = MatrixAdaptable;
    using EigenVec_t = Parent::EigenVecRef;
    using EigenCVec_t = Parent::EigenCVecRef;
    //! Default constructor
    DenseEigenAdaptor() = default;

    //! constructor from an Eigen matrix
    explicit DenseEigenAdaptor(const Eigen::Ref<const Eigen::MatrixXd> matrix);

    //! constructor creates a zero matrix
    explicit DenseEigenAdaptor(const Index_t & nb_dof);

    //! Copy constructor
    DenseEigenAdaptor(const DenseEigenAdaptor & other) = default;

    //! Move constructor
    DenseEigenAdaptor(DenseEigenAdaptor && other) = default;

    //! Destructor
    virtual ~DenseEigenAdaptor() = default;

    //! Copy assignment operator
    DenseEigenAdaptor & operator=(const DenseEigenAdaptor & other) = delete;

    //! Move assignment operator
    DenseEigenAdaptor & operator=(DenseEigenAdaptor && other) = delete;

    Index_t get_nb_dof() const final;

    void action_increment(EigenCVec_t delta_grad, const Real & alpha,
                          EigenVec_t del_flux) final;
    const muGrid::Communicator & get_communicator() const final;

    Eigen::MatrixXd & get_matrix();
    const Eigen::MatrixXd & get_matrix() const;

   protected:
    Eigen::MatrixXd matrix;
    muGrid::Communicator comm{muGrid::Communicator()};
  };

}  // namespace muSpectre

namespace Eigen {
  namespace internal {
    //! Implementation of `muSpectre::MatrixAdaptor` * `Eigen::DenseVector`
    //! through a specialization of `Eigen::internal::generic_product_impl`:
    template <typename Rhs>  // GEMV stands for matrix-vector
    struct generic_product_impl<muSpectre::MatrixAdaptor, Rhs, SparseShape,
                                DenseShape, GemvProduct>
        : generic_product_impl_base<
              muSpectre::MatrixAdaptor, Rhs,
              generic_product_impl<muSpectre::MatrixAdaptor, Rhs>> {
      //! undocumented
      typedef typename Product<muSpectre::MatrixAdaptor, Rhs>::Scalar Scalar;

      //! undocumented
      template <typename Dest>
      static void scaleAndAddTo(Dest & dst,
                                const muSpectre::MatrixAdaptor & lhs,
                                const Rhs & rhs, const Scalar & alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so
        // let's not bother about it.
        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        lhs.action_increment(rhs, alpha, dst);
      }
    };
  }  // namespace internal
}  // namespace Eigen

#endif  // SRC_SOLVER_MATRIX_ADAPTOR_HH_
