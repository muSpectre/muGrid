/**
 * @file   test_goodies.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Sep 2017
 *
 * @brief  helpers for testing
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
 * General Public License for more details.
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
 */

#ifndef TESTS_TEST_GOODIES_HH_
#define TESTS_TEST_GOODIES_HH_

#include "common/tensor_algebra.hh"

#include <boost/mpl/list.hpp>

#include <random>
#include <type_traits>

namespace muSpectre {

  namespace testGoodies {

    template <Dim_t Dim> struct dimFixture { constexpr static Dim_t dim{Dim}; };

    using dimlist = boost::mpl::list<dimFixture<oneD>, dimFixture<twoD>,
                                     dimFixture<threeD>>;

    /* ---------------------------------------------------------------------- */
    template <typename T> class RandRange {
     public:
      RandRange() : rd(), gen(rd()) {}

      template <typename dummyT = T>
      std::enable_if_t<std::is_floating_point<dummyT>::value, dummyT>
      randval(T &&lower, T &&upper) {
        static_assert(std::is_same<T, dummyT>::value, "SFINAE");
        auto distro = std::uniform_real_distribution<T>(lower, upper);
        return distro(this->gen);
      }

      template <typename dummyT = T>
      std::enable_if_t<std::is_integral<dummyT>::value, dummyT>
      randval(T &&lower, T &&upper) {
        static_assert(std::is_same<T, dummyT>::value, "SFINAE");
        auto distro = std::uniform_int_distribution<T>(lower, upper);
        return distro(this->gen);
      }

     private:
      std::random_device rd;
      std::default_random_engine gen;
    };

    /**
     * explicit computation of linearisation of PK1 stress for an
     * objective Hooke's law. This implementation is not meant to be
     * efficient, but te reflect exactly the formulation in Curnier
     * 2000, "Méthodes numériques en mécanique des solides" for
     * reference and testing
     */
    template <Dim_t Dim>
    decltype(auto) objective_hooke_explicit(Real lambda, Real mu,
                                            const Matrices::Tens2_t<Dim> &F) {
      using Matrices::tensmult;
      using Matrices::Tens4_t;
      using Matrices::Tens2_t;

      using T2 = Tens2_t<Dim>;
      using T4 = Tens4_t<Dim>;
      T2 P;
      T2 I = P.Identity();
      T4 K;
      // See Curnier, 2000, "Méthodes numériques en mécanique des
      // solides", p 252, (6.95b)
      Real Fjrjr = (F.array() * F.array()).sum();
      T2 Fjrjm = F.transpose() * F;
      P.setZero();
      for (Dim_t i = 0; i < Dim; ++i) {
        for (Dim_t m = 0; m < Dim; ++m) {
          P(i, m) += lambda / 2 * (Fjrjr - Dim) * F(i, m);
          for (Dim_t r = 0; r < Dim; ++r) {
            P(i, m) += mu * F(i, r) * (Fjrjm(r, m) - I(r, m));
          }
        }
      }
      // See Curnier, 2000, "Méthodes numériques en mécanique des solides", p
      // 252
      Real Fkrkr = (F.array() * F.array()).sum();
      T2 Fkmkn = F.transpose() * F;
      T2 Fisjs = F * F.transpose();
      K.setZero();
      for (Dim_t i = 0; i < Dim; ++i) {
        for (Dim_t j = 0; j < Dim; ++j) {
          for (Dim_t m = 0; m < Dim; ++m) {
            for (Dim_t n = 0; n < Dim; ++n) {
              get(K, i, m, j, n) =
                  (lambda * ((Fkrkr - Dim) / 2 * I(i, j) * I(m, n) +
                             F(i, m) * F(j, n)) +
                   mu * (I(i, j) * Fkmkn(m, n) + Fisjs(i, j) * I(m, n) -
                         I(i, j) * I(m, n) + F(i, n) * F(j, m)));
            }
          }
        }
      }
      return std::make_tuple(P, K);
    }

  }  // namespace testGoodies

}  // namespace muSpectre

#endif  // TESTS_TEST_GOODIES_HH_
