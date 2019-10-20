/**
 * @file   material_linear_anisotropic.cc
 *
 * @author Ali Falsafi<ali.falsafi@epfl.ch>
 *
 * @date  09 Jul 2018
 *
 * @brief  Implementation of general anisotropic linear constitutive model
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

#include "material_linear_anisotropic.hh"

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearAnisotropic<DimS, DimM>::MaterialLinearAnisotropic(
      std::string name, std::vector<Real> input_c)
      : Parent(name), C{c_maker(input_c)} {};

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialLinearAnisotropic<DimS, DimM>::c_maker(std::vector<Real> input)
      -> Stiffness_t {
    // the correct size of the input according to the dimension
    std::array<Dim_t, 2> constexpr input_size{6, 21};
    // stiffness_matrix
    Stiffness_t C4{Stiffness_t::Zero()};
    // voigt stiffness matrix
    Stiffness_t C4_v{Stiffness_t::Zero()};

    if (input.size() != size_t(input_size[DimM - 2])) {
      std::stringstream err_str{};
      err_str << "Number of the inputs should be" << input_size[DimM - 2]
              << std::endl;
      throw std::runtime_error(err_str.str());
    }

    constexpr Dim_t v_diff{DimM * DimM - vsize(DimM)};

    // memory order voigt -> col major
    using t_t = VoigtConversion<DimM>;
    auto v_order{t_t::get_vec_vec()};  // non-sym for C

    int counter{0};

    for (int i{0}; i < vsize(DimM); ++i) {
      // diagonal terms
      C4(i, i) = input[counter];
      counter++;
      for (int j{i + 1}; j < vsize(DimM); ++j) {
        C4(i, j) = C4(j, i) = input[counter];
        if (j >= DimM) {
          C4(j + v_diff, i) = C4(j, i + v_diff) = C4(i + v_diff, j) =
              C4(i, j + v_diff) = input[counter];
        }
        counter++;
      }
    }

    C4.bottomRightCorner(v_diff, v_diff) =
        C4.block(DimM + v_diff, DimM, v_diff, v_diff) =
            C4.block(DimM, DimM + v_diff, v_diff, v_diff) =
                C4.block(DimM, DimM, v_diff, v_diff);

    for (int i = 0; i < DimM * DimM; i++) {
      for (int j = 0; j < DimM * DimM; j++) {
        C4_v(i, j) = C4(v_order[i], v_order[j]);
      }
    }
    return C4_v;
  }

  /* ---------------------------------------------------------------------- */

  template class MaterialLinearAnisotropic<twoD, twoD>;
  template class MaterialLinearAnisotropic<twoD, threeD>;
  template class MaterialLinearAnisotropic<threeD, threeD>;

}  // namespace muSpectre
