/**
 * @file   test_fields.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test Fields that are used in FieldCollections
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

#include "tests.hh"
#include "common/field_collection.hh"
#include "common/field.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_test);

  BOOST_AUTO_TEST_CASE(simple_creation) {
    const Dim_t sdim = 2;
    const Dim_t mdim = 2;
    const Dim_t order = 4;
    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc;

    using TF_t = TensorField<FC_t, Real, order, mdim>;
    make_field<TF_t>("TensorField 1", fc);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
