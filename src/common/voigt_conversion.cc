/**
 * @file   voigt_conversion.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2017
 *
 * @brief  specializations for static members of voigt converter
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */


#include "common/voigt_conversion.hh"

namespace muSpectre {

  //! voigt vector indices for non-symmetric tensors
  template <>
  const Eigen::Matrix<Dim_t, 1, 1>
      VoigtConversion<1>::mat = (Eigen::Matrix<Dim_t, 1, 1>() << 0).finished();
  //! voigt vector indices for non-symmetric tensors
  template <>
  const Eigen::Matrix<Dim_t, 2, 2> VoigtConversion<2>::mat =
      (Eigen::Matrix<Dim_t, 2, 2>() << 0, 2, 3, 1).finished();
  //! voigt vector indices for non-symmetric tensors
  template <>
  const Eigen::Matrix<Dim_t, 3, 3> VoigtConversion<3>::mat =
      (Eigen::Matrix<Dim_t, 3, 3>() << 0, 5, 4, 8, 1, 3, 7, 6, 2).finished();
  //! voigt vector indices
  template <>
  const Eigen::Matrix<Dim_t, 1, 1> VoigtConversion<1>::sym_mat =
      (Eigen::Matrix<Dim_t, 1, 1>() << 0).finished();
  //! voigt vector indices
  template <>
  const Eigen::Matrix<Dim_t, 2, 2> VoigtConversion<2>::sym_mat =
      (Eigen::Matrix<Dim_t, 2, 2>() << 0, 2, 2, 1).finished();
  //! voigt vector indices
  template <>
  const Eigen::Matrix<Dim_t, 3, 3> VoigtConversion<3>::sym_mat =
      (Eigen::Matrix<Dim_t, 3, 3>() << 0, 5, 4, 5, 1, 3, 4, 3, 2).finished();
  //! matrix indices from voigt vectors
  template <>
  const Eigen::Matrix<Dim_t, 1 * 1, 2> VoigtConversion<1>::vec =
      (Eigen::Matrix<Dim_t, 1 * 1, 2>() << 0, 0).finished();
  //! matrix indices from voigt vectors
  template <>
  const Eigen::Matrix<Dim_t, 2 * 2, 2> VoigtConversion<2>::vec =
      (Eigen::Matrix<Dim_t, 2 * 2, 2>() << 0, 0, 1, 1, 0, 1, 1, 0).finished();
  //! matrix indices from voigt vectors
  template <>
  const Eigen::Matrix<Dim_t, 3 * 3, 2>
      VoigtConversion<3>::vec = (Eigen::Matrix<Dim_t, 3 * 3, 2>() << 0, 0, 1, 1,
                                 2, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0)
                                    .finished();
  //! factors for shear components in voigt notation
  template <>
  const Eigen::Matrix<Real, vsize(1), 1> VoigtConversion<1>::factors =
      (Eigen::Matrix<Real, vsize(1), 1>() << 1).finished();
  //! factors for shear components in voigt notation
  template <>
  const Eigen::Matrix<Real, vsize(2), 1> VoigtConversion<2>::factors =
      (Eigen::Matrix<Real, vsize(2), 1>() << 1, 1, 2).finished();
  //! factors for shear components in voigt notation
  template <>
  const Eigen::Matrix<Real, vsize(3), 1> VoigtConversion<3>::factors =
      (Eigen::Matrix<Real, vsize(3), 1>() << 1, 1, 1, 2, 2, 2).finished();

  /**
   * reordering between a row/column in voigt vs col-major matrix
   * (e.g., stiffness tensor)
   */
  template <>
  const Eigen::Matrix<Dim_t, 1 * 1, 1> VoigtConversion<1>::vec_vec{
      (Eigen::Matrix<Dim_t, 1 * 1, 1>() << 0).finished()};
  template <>
  const Eigen::Matrix<Dim_t, 2 * 2, 1> VoigtConversion<2>::vec_vec{
      (Eigen::Matrix<Dim_t, 2 * 2, 1>() << 0, 3, 2, 1).finished()};
  template <>
  const Eigen::Matrix<Dim_t, 3 * 3, 1> VoigtConversion<3>::vec_vec{
      (Eigen::Matrix<Dim_t, 3 * 3, 1>() << 0, 8, 7, 5, 1, 6, 4, 3, 2)
          .finished()};

  template class VoigtConversion<1>;
  template class VoigtConversion<2>;
  template class VoigtConversion<3>;

}  // namespace muSpectre
