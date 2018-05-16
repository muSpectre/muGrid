/**
 * @file   test_field_collections.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   23 Nov 2017
 *
 * @brief  declares fixtures for field_collection tests, so that they can be split
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

#ifndef TEST_FIELD_COLLECTIONS_H
#define TEST_FIELD_COLLECTIONS_H
#include <stdexcept>
#include <boost/mpl/list.hpp>
#include <random>
#include <type_traits>
#include <sstream>
#include <string>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "tests/test_goodies.hh"
#include "tests.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common/field_map.hh"

namespace muSpectre {

  //! Test fixture for simple tests on single field in collection
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_fixture:
    public std::conditional_t<Global,
                              GlobalFieldCollection<DimS>,
                              LocalFieldCollection<DimS>> {
    FC_fixture()
      :fc() {}
    inline static constexpr Dim_t sdim(){return DimS;}
    inline static constexpr Dim_t mdim(){return DimM;}
    inline static constexpr bool global(){return Global;}
    using FC_t = std::conditional_t<Global,
                                    GlobalFieldCollection<DimS>,
                                    LocalFieldCollection<DimS>>;
    FC_t fc;
  };

  using test_collections = boost::mpl::list<FC_fixture<2, 2, true>,
                                            FC_fixture<2, 3, true>,
                                            FC_fixture<3, 3, true>,
                                            FC_fixture<2, 2, false>,
                                            FC_fixture<2, 3, false>,
                                            FC_fixture<3, 3, false>>;

  constexpr Dim_t order{4}, matrix_order{2};
  //! Test fixture for multiple fields in the collection
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_multi_fixture{
    using FC_t = std::conditional_t<Global,
                                   GlobalFieldCollection<DimS>,
                                   LocalFieldCollection<DimS>>;
    using T4_t = TensorField<FC_t, Real, order, DimM>;
    using T2_t = TensorField<FC_t, Real, matrix_order, DimM>;
    using Sc_t = ScalarField<FC_t, Int>;
    using M2_t = MatrixField<FC_t, Complex, DimS, DimM>;
    using Dyn_t = TypedField<FC_t, Real>;

    FC_multi_fixture()
      :fc(),
       t4_field{make_field<T4_t>("Tensorfield Real o4", fc)},//Real tensor field
       t2_field{make_field<T2_t>("Tensorfield Real o2", fc)},//Real tensor field
       sc_field{make_field<Sc_t>("integer Scalar", fc)}, // integer scalar field
       m2_field{make_field<M2_t>("Matrixfield Complex sdim x mdim", fc)}, //complex matrix field
       dyn_field{make_field<Dyn_t>("Dynamically sized Field", fc, 12)}
    {
    }
    inline static constexpr Dim_t sdim(){return DimS;}
    inline static constexpr Dim_t mdim(){return DimM;}
    inline static constexpr bool global(){return Global;}
    FC_t fc;
    T4_t & t4_field;
    T2_t & t2_field;
    Sc_t & sc_field;
    M2_t & m2_field;
    Dyn_t & dyn_field;
  };

  using mult_collections = boost::mpl::list<FC_multi_fixture<2, 2, true>,
                                            FC_multi_fixture<2, 3, true>,
                                            FC_multi_fixture<3, 3, true>,
                                            FC_multi_fixture<2, 2, false>,
                                            FC_multi_fixture<2, 3, false>,
                                            FC_multi_fixture<3, 3, false>>;

  //! Test fixture for iterators over multiple fields
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_iterator_fixture
    : public FC_multi_fixture<DimS, DimM, Global> {
    using Parent = FC_multi_fixture<DimS, DimM, Global>;
    FC_iterator_fixture()
      :Parent() {
      this-> fill();
    }

    template <bool isGlobal = Global>
    std::enable_if_t<isGlobal> fill() {
      static_assert(Global==isGlobal, "You're breaking my SFINAE plan");
      Ccoord_t<Parent::sdim()> size;
      Ccoord_t<Parent::sdim()> loc{};
      for (auto && s: size) {
        s = cube_size();
      }
      this->fc.initialise(size, loc);
    }

    template <bool notGlobal = !Global>
    std::enable_if_t<notGlobal> fill (int dummy = 0) {
      static_assert(notGlobal != Global, "You're breaking my SFINAE plan");
      testGoodies::RandRange<Int> rng;
      this->fc.add_pixel({0,0});
      for (int i = 0*dummy; i < sele_size(); ++i) {
        Ccoord_t<Parent::sdim()> pixel;
        for (auto && s: pixel) {
          s = rng.randval(0, 7);
        }
        this->fc.add_pixel(pixel);
      }

      this->fc.initialise();
    }

    constexpr static Dim_t cube_size() {return 3;}
    constexpr static Dim_t sele_size() {return 7;}
  };

  using iter_collections = boost::mpl::list<FC_iterator_fixture<2, 2, true>,
                                            FC_iterator_fixture<2, 3, true>,
                                            FC_iterator_fixture<3, 3, true>,
                                            FC_iterator_fixture<2, 2, false>,
                                            FC_iterator_fixture<2, 3, false>,
                                            FC_iterator_fixture<3, 3, false>>;

  using glob_iter_colls = boost::mpl::list<FC_iterator_fixture<2, 2, true>,
                                           FC_iterator_fixture<2, 3, true>,
                                           FC_iterator_fixture<3, 3, true>>;

}  // muSpectre


#endif /* TEST_FIELD_COLLECTIONS_H */
