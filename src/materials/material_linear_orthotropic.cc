/**
 * @file   material_linear_orthotropic.cc
 *
 * @author Ali Falsafi<ali.falsafi@epfl.ch>
 *
 * @date  11 Jul 2018
 *
 * @brief  Implementation of general orthotropic linear constitutive model
 *
 * Copyright © 2017 Till Junge
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

#include "material_base.hh"
#include "common/muSpectre_common.hh"
#include "material_linear_anisotropic.hh"
#include "material_linear_orthotropic.hh"

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearOrthotropic<DimS, DimM>::MaterialLinearOrthotropic(
      std::string name, std::vector<Real> input)
      : Parent(name, input_c_maker(input)) {}

  template <Dim_t DimS, Dim_t DimM>
  std::vector<Real> MaterialLinearOrthotropic<DimS, DimM>::input_c_maker(
      std::vector<Real> input) {
    std::array<Dim_t, 2> constexpr input_size{4, 9};
    std::array<Dim_t, 2> constexpr output_size{6, 21};
    std::vector<Real> retval{};
    // in case the length of the input is inconsistnent:
    if (input.size() != size_t(input_size[DimM - 2])) {
      std::stringstream err_str{};
      err_str << "Number of the inputs should be" << input_size[DimM - 2]
              << std::endl;
      throw std::runtime_error(err_str.str());
    }
    Dim_t S{output_size[DimS - 2]};
    Dim_t counter{0};
    for (Dim_t i = 0; i < S; ++i) {
      if (this->ret_flag[i]) {
        retval.push_back(input[counter]);
        counter++;
      } else {
        retval.push_back(0.0);
      }
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <>
  std::array<bool, 6> MaterialLinearOrthotropic<twoD, twoD>::ret_flag = {
      1, 1, 0, 1, 0, 1};

  template <>
  std::array<bool, 21> MaterialLinearOrthotropic<threeD, threeD>::ret_flag = {
      1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1};

  /* ---------------------------------------------------------------------- */
  template class MaterialLinearOrthotropic<twoD, twoD>;
  // template class MaterialLinearOrthotropic<twoD, threeD>;
  template class MaterialLinearOrthotropic<threeD, threeD>;

}  // namespace muSpectre
