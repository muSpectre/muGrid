/**
 * file   solver_error.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   28 Dec 2017
 *
 * @brief  Errors raised by solvers
 *
 * @section LICENCE
 *
 * Copyright © 2017 Till Junge
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

#ifndef SOLVER_ERROR_H
#define SOLVER_ERROR_H

#include <stdexcept>

namespace muSpectre {

  class SolverError: public std::runtime_error {
    using runtime_error::runtime_error;
  };

  class ConvergenceError: public SolverError {
    using SolverError::SolverError;
  };

}  // muSpectre


#endif /* SOLVER_ERROR_H */
