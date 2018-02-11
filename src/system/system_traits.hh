/**
 * @file   system_traits.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Jan 2018
 *
 * @brief  Provides traits for Eigen solvers to be able to use Systems
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "common/common.hh"

#include <Eigen/IterativeLinearSolvers>

namespace muSpectre {

  template <class System>
  class SystemAdaptor;

}  // muSpectre

namespace Eigen {
  namespace internal {
    using Dim_t = muSpectre::Dim_t; //!< universal index type
    using Real =  muSpectre::Real;  //!< universal real value type
    template<class System>
    struct traits<muSpectre::SystemAdaptor<System>> :
      public Eigen::internal::traits<Eigen::SparseMatrix<Real> >
    {};
  }  // internal
}  // Eigen
