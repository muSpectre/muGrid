/**
 * @file   mufft_common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Small definitions of commonly used types throughout µFFT
 *
 * Copyright © 2019 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#include <libmugrid/grid_common.hh>

#ifndef SRC_LIBMUFFT_MUFFT_COMMON_HH_
#define SRC_LIBMUFFT_MUFFT_COMMON_HH_

namespace muFFT {
  using muGrid::Dim_t;
  using muGrid::Index_t;

  using muGrid::Complex;
  using muGrid::Int;
  using muGrid::Real;
  using muGrid::Uint;

  using muGrid::RuntimeError;

  using muGrid::Ccoord_t;
  using muGrid::DynCcoord_t;
  using muGrid::DynRcoord_t;
  using muGrid::Rcoord_t;
  using muGrid::Shape_t;

  using muGrid::optional;

  using muGrid::oneD;
  using muGrid::threeD;
  using muGrid::twoD;

  using muGrid::OneQuadPt;
  using muGrid::TwoQuadPts;
  using muGrid::FourQuadPts;
  using muGrid::OneNode;

  using muGrid::Mapping;
  using muGrid::IterUnit;

  /**
   * Planner flags for FFT (follows FFTW, hopefully this choice will
   * be compatible with alternative FFT implementations)
   * @enum muFFT::FFT_PlanFlags
   */
  enum class FFT_PlanFlags {
    estimate,  //!< cheapest plan for slowest execution
    measure,   //!< more expensive plan for fast execution
    patient    //!< very expensive plan for fastest execution
  };

  /**
   * used to tag all fields (libµgrid allows for pixel-sub-divisions, which
   * libµFFt does not use.
   */
  using muGrid::PixelTag;

  /**
   * used by the lazy-evaluated FFTExpression class ;
   */
  enum class FFTDirection { Forward, Inverse };

  namespace version {

    /**
     * returns a formatted text that can be printed to stdout or to output
     * files. It contains the git commit hash and repository url used to compile
     * µSpectre and whether the current state was dirty or not.
     */
    std::string info();
    const char * hash();
    const char * description();
    bool is_dirty();

  }  // namespace version

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_MUFFT_COMMON_HH_
