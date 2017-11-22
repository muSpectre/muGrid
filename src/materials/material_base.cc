/**
 * file   material_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  implementation of materi
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

#include "materials/material_base.hh"

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template<Dim_t s_dim, Dim_t m_dim>
  MaterialBase<s_dim, m_dim>::MaterialBase(std::string name)
    :name(name) {
    static_assert((m_dim == oneD)||
                  (m_dim == twoD)||
                  (m_dim == threeD), "only 1, 2, or threeD supported");
  }

  template class MaterialBase<2, 2>;
  template class MaterialBase<2, 3>;
  template class MaterialBase<3, 3>;

}  // muSpectre
