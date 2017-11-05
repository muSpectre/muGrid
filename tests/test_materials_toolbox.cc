/**
 * file   test_materials_toolbox.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Tests for the materials toolbox
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

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/TensorSymmetry>

#include "tests.hh"
#include "materials/materials_toolbox.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(materials_toolbox)

  BOOST_AUTO_TEST_CASE(test_linearisation) {
    constexpr Dim_t dim{2};
    using Stress_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>>;
    using Strain_t = Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim>>;
    Strain_t F;
    F.setRandom();
    auto E = .5*(F.shuffle(std::array<Dim_t, dim>{1,0})*F-Tensors::I2<dim>());
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
