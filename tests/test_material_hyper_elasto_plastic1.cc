/**
 * @file   test_material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   25 Feb 2018
 *
 * @brief  Tests for the large-strain Simo-type plastic law implemented 
 *         using MaterialMuSpectre
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

#include "boost/mpl/list.hpp"

#include "materials/material_hyper_elasto_plastic1.hh"
#include "tests.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elasto-plastic_1);

  template <class Mat_t>
  struct MaterialFixture
  {
    using Mat = Mat_t;
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real young{mu*(3*lambda + 2*mu)/(lambda + mu)};
    constexpr static Real poisson{lambda/(2*(lambda + mu))};
    MaterialFixture():mat("Name", young, poisson){};
    constexpr static Dim_t sdim{Mat_t::sdim()};
    constexpr static Dim_t mdim{Mat_t::mdim()};

    Mat_t mat;
  };

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
