/**
 * file   ccoord_operations.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  common operations on pixel addressing 
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <functional>
#include <numeric>

#include "common/common.hh"

#ifndef CCOORD_OPERATIONS_H
#define CCOORD_OPERATIONS_H

namespace muSpectre {

  namespace CcoordOps {

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr Ccoord_t<dim> get_ccoord(const Ccoord_t<dim> & sizes, Dim_t index) {
      Ccoord_t<dim> retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval[i] = index/factor%sizes[i];
        if (dim != 0 ) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr Dim_t get_index(const Ccoord_t<dim> & sizes,
                              const Ccoord_t<dim> & ccoord) {
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval += ccoord[i]*factor;
        if (dim != 0) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr size_t get_size(const Ccoord_t<dim>& sizes) {
      return std::accumulate(sizes.begin(), sizes.end(), 1,
                             std::multiplies<size_t>());
    }



  }  // CcoordOps

}  // muSpectre

#endif /* CCOORD_OPERATIONS_H */
