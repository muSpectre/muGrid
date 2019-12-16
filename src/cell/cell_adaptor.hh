/**
 * @file   cell_adaptor.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Sep 2019
 *
 * @brief  Cell Adaptor implements the matrix-vector multiplication and allows
 *         the adapted cell to be used like a spacse matrix in
 *         conjugate-gradient-type solvers
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_CELL_CELL_ADAPTOR_HH_
#define SRC_CELL_CELL_ADAPTOR_HH_

#include <common/muSpectre_common.hh>

#include <Eigen/IterativeLinearSolvers>

namespace muSpectre {

  template <class Cell>
  class CellAdaptor;

}  // namespace muSpectre

namespace Eigen {
  namespace internal {
    using Dim_t = muSpectre::Dim_t;  //!< universal index type
    using Real = muSpectre::Real;    //!< universal real value type
    template <class Cell>
    struct traits<muSpectre::CellAdaptor<Cell>>
        : public Eigen::internal::traits<Eigen::SparseMatrix<Real>> {};
  }  // namespace internal
}  // namespace Eigen

namespace muSpectre {

  /**
   * lightweight resource handle wrapping a `muSpectre::Cell` or
   * a subclass thereof into `Eigen::EigenBase`, so it can be
   * interpreted as a sparse matrix by Eigen solvers
   */
  template <class Cell>
  class CellAdaptor : public Eigen::EigenBase<CellAdaptor<Cell>> {
   public:
    using Scalar = double;      //!< sparse matrix traits
    using RealScalar = double;  //!< sparse matrix traits
    using StorageIndex = int;   //!< sparse matrix traits
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      RowsAtCompileTime = Eigen::Dynamic,
      MaxRowsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
    };

    //! constructor
    explicit CellAdaptor(Cell & cell) : cell{cell} {}
    //! returns the number of logical rows
    Eigen::Index rows() const { return this->cell.get_nb_dof(); }
    //! returns the number of logical columns
    Eigen::Index cols() const { return this->rows(); }

    //! implementation of the evaluation
    template <typename Rhs>
    Eigen::Product<CellAdaptor, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs> & x) const {
      return Eigen::Product<CellAdaptor, Rhs, Eigen::AliasFreeProduct>(
          *this, x.derived());
    }
    Cell & cell;  //!< ref to the cell
  };
}  // namespace muSpectre

namespace Eigen {
  namespace internal {
    //! Implementation of `muSpectre::CellAdaptor` * `Eigen::DenseVector`
    //! through a specialization of `Eigen::internal::generic_product_impl`:
    template <typename Rhs, class CellAdaptor>  // GEMV stands for matrix-vector
    struct generic_product_impl<CellAdaptor, Rhs, SparseShape, DenseShape,
                                GemvProduct>
        : generic_product_impl_base<CellAdaptor, Rhs,
                                    generic_product_impl<CellAdaptor, Rhs>> {
      //! undocumented
      typedef typename Product<CellAdaptor, Rhs>::Scalar Scalar;

      //! undocumented
      template <typename Dest>
      static void scaleAndAddTo(Dest & dst, const CellAdaptor & lhs,
                                const Rhs & rhs, const Scalar & alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so
        // let's not bother about it.
        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        auto & cell{const_cast<CellAdaptor &>(lhs).cell};
        cell.add_projected_directional_stiffness(rhs, alpha, dst);
      }
    };
  }  // namespace internal
}  // namespace Eigen

#endif  // SRC_CELL_CELL_ADAPTOR_HH_
