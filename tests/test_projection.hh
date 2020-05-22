/**
 * @file   test_projection.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   16 Jan 2018
 *
 * @brief  common declarations for testing both the small and finite strain
 *         projection operators
 *
 * Copyright © 2018 Till Junge
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

#include "tests.hh"
#include <libmufft/derivative.hh>
#include <libmufft/fftw_engine.hh>

#include <boost/mpl/list.hpp>
#include <Eigen/Dense>
#include <iostream>

#ifndef TESTS_TEST_PROJECTION_HH_
#define TESTS_TEST_PROJECTION_HH_

using muFFT::Gradient_t;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  struct Sizes {};
  template <>
  struct Sizes<twoD> {
    constexpr static Ccoord_t<twoD> get_nb_grid_pts() {
      return Ccoord_t<twoD>{3, 5};
    }
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{3.4, 5.8};
    }
  };
  template <>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_nb_grid_pts() {
      return Ccoord_t<threeD>{3, 5, 7};
    }
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{3.4, 5.8, 6.7};
    }
  };
  template <Dim_t DimS>
  struct Squares {};
  template <>
  struct Squares<twoD> {
    constexpr static Ccoord_t<twoD> get_nb_grid_pts() {
      return Ccoord_t<twoD>{5, 5};
    }
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{5, 5};
    }
  };
  template <>
  struct Squares<threeD> {
    constexpr static Ccoord_t<threeD> get_nb_grid_pts() {
      return Ccoord_t<threeD>{7, 7, 7};
    }
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{7, 7, 7};
    }
  };
  template <Dim_t DimS>
  struct FourierGradient {};
  template <>
  struct FourierGradient<twoD> {
    static Gradient_t get_gradient() {
      return Gradient_t{std::make_shared<muFFT::FourierDerivative>(twoD, 0),
                        std::make_shared<muFFT::FourierDerivative>(twoD, 1)};
    }
  };
  template <>
  struct FourierGradient<threeD> {
    static Gradient_t get_gradient() {
      return Gradient_t{std::make_shared<muFFT::FourierDerivative>(threeD, 0),
                        std::make_shared<muFFT::FourierDerivative>(threeD, 1),
                        std::make_shared<muFFT::FourierDerivative>(threeD, 2)};
    }
  };
  template <Dim_t DimS, Dim_t NbQuadPts = OneQuadPt>
  struct DiscreteGradient {};
  template <>
  struct DiscreteGradient<twoD, OneQuadPt> {
    static Gradient_t get_gradient() {
      return Gradient_t{std::make_shared<muFFT::DiscreteDerivative>(
                            DynCcoord_t{2, 2}, DynCcoord_t{0, 0},
                            std::vector<Real>{-0.5, -0.5, 0.5, 0.5}),
                        std::make_shared<muFFT::DiscreteDerivative>(
                            DynCcoord_t{2, 2}, DynCcoord_t{0, 0},
                            std::vector<Real>{-0.5, 0.5, -0.5, 0.5})};
    }
  };
  template <>
  struct DiscreteGradient<twoD, TwoQuadPts> {
    static Gradient_t get_gradient() {
      return Gradient_t{
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 1}, DynCcoord_t{0, 0}, std::vector<Real>{-1, 1}),
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{1, 2}, DynCcoord_t{0, 0}, std::vector<Real>{-1, 1}),
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 2}, DynCcoord_t{0, 0},
              std::vector<Real>{0, 0, -1, 1}),
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 2}, DynCcoord_t{0, 0},
              std::vector<Real>{0, -1, 0, 1})};
    }
  };
  template <>
  struct DiscreteGradient<threeD, OneQuadPt> {
    static Gradient_t get_gradient() {
      return Gradient_t{
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 2, 2}, DynCcoord_t{0, 0, 0},
              std::vector<Real>{-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5}),
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 2, 2}, DynCcoord_t{0, 0, 0},
              std::vector<Real>{-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5}),
          std::make_shared<muFFT::DiscreteDerivative>(
              DynCcoord_t{2, 2, 2}, DynCcoord_t{0, 0, 0},
              std::vector<Real>{-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5})};
    }
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class SizeGiver_, class GradientGiver_,
            class Proj, Dim_t NbQuadPts = 1>
  struct ProjectionFixture {
    using Engine = muFFT::FFTWEngine;
    using Parent = Proj;
    using SizeGiver = SizeGiver_;
    using GradientGiver = GradientGiver_;
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    constexpr static Dim_t nb_quad{NbQuadPts};
    ProjectionFixture()
        : projector(std::make_unique<Engine>(
                        DynCcoord_t(SizeGiver::get_nb_grid_pts())),
                    DynRcoord_t(SizeGiver::get_lengths()),
                    Gradient_t(GradientGiver::get_gradient())) {}
    Parent projector;
  };

}  // namespace muSpectre

#endif  // TESTS_TEST_PROJECTION_HH_
