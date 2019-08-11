/**
 * @file   test_nfield_collection.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Tests for FieldCollection
 *
 * Copyright © 2019 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "tests.hh"

#include "libmugrid/nfield_collection_global.hh"
#include "libmugrid/nfield_collection_local.hh"
#include "libmugrid/nfield_typed.hh"
#include "libmugrid/iterators.hh"
#include "libmugrid/ccoord_operations.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(nfield_collection_test);

  template <Dim_t DimS>
  struct GlobalFCFixture {
    constexpr static Dim_t spatial_dimension() { return DimS; }
    GlobalNFieldCollection<DimS> fc{Unknown};
  };

  using GlobalFCFixtures =
      boost::mpl::list<GlobalFCFixture<twoD>, GlobalFCFixture<threeD>>;

  struct LocalFCFixture {
    LocalNFieldCollection fc{Unknown, Unknown};
  };

  BOOST_AUTO_TEST_CASE(NFieldCollection_construction) {
    BOOST_CHECK_NO_THROW(LocalNFieldCollection(Unknown, Unknown));
    using GlobalNFieldCollection2d = GlobalNFieldCollection<twoD>;
    using GlobalNFieldCollection3d = GlobalNFieldCollection<threeD>;
    BOOST_CHECK_NO_THROW(GlobalNFieldCollection2d{Unknown});
    BOOST_CHECK_NO_THROW(GlobalNFieldCollection3d{Unknown});
  }

  // the following test only tests members of the NFieldCollection base class,
  // so it is not necessary to test it on both LocalNFieldCollection
  // GlobalNFieldCollection
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(registration_test, Fix, GlobalFCFixtures,
                                   Fix) {
    BOOST_CHECK(Fix::fc.get_domain() == NFieldCollection::Domain::Global);
    const std::string right_name{"right name"}, wrong_name{"wrong name"};
    Fix::fc.template register_field<TypedNField<Real>>(right_name, 1);
    const bool should_be_true{Fix::fc.field_exists(right_name)};
    const bool should_be_false{Fix::fc.field_exists(wrong_name)};
    BOOST_CHECK_EQUAL(true, should_be_true);
    BOOST_CHECK_EQUAL(false, should_be_false);
    BOOST_CHECK_EQUAL(Unknown, Fix::fc.size());
    BOOST_CHECK_THROW(Fix::fc.template register_field<TypedNField<Real>>(
                          right_name, 24),
                      NFieldCollectionError);
  }

  // the following test only tests members of the NFieldCollection base class,
  // so it is not necessary to test it on both LocalNFieldCollection
  // GlobalNFieldCollection
  BOOST_AUTO_TEST_CASE(local_registration_test) {
    LocalNFieldCollection fc{Unknown, Unknown};
    BOOST_CHECK(fc.get_domain() == NFieldCollection::Domain::Local);
    const std::string right_name{"right name"}, wrong_name{"wrong name"};
    fc.template register_field<TypedNField<Real>>(right_name, 1);
    const bool should_be_true{fc.field_exists(right_name)};
    const bool should_be_false{fc.field_exists(wrong_name)};
    BOOST_CHECK_EQUAL(true, should_be_true);
    BOOST_CHECK_EQUAL(false, should_be_false);
    BOOST_CHECK_EQUAL(Unknown, fc.size());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(initialisation_test_global, Fix,
                                   GlobalFCFixtures, Fix) {
    Ccoord_t<Fix::spatial_dimension()> nb_grid_pts{};
    Dim_t nb_pixels{1};
    constexpr Dim_t NbQuad{3};
    for (int i{0}; i < Fix::spatial_dimension(); ++i) {
      const auto nb_grid{2 * i + 1};
      nb_grid_pts[i] = nb_grid;
      nb_pixels *= nb_grid;
    }
    CcoordOps::Pixels<Fix::spatial_dimension()> pixels{nb_grid_pts};
    BOOST_CHECK_THROW(Fix::fc.initialise(nb_grid_pts), NFieldCollectionError);
    Fix::fc.set_nb_quad(NbQuad);
    BOOST_CHECK(not Fix::fc.is_initialised());
    BOOST_CHECK_NO_THROW(Fix::fc.initialise(nb_grid_pts));
    BOOST_CHECK(Fix::fc.is_initialised());
    BOOST_CHECK_EQUAL(nb_pixels * NbQuad, Fix::fc.size());

    //! double initialisation is forbidden
    BOOST_CHECK_THROW(Fix::fc.initialise(nb_grid_pts), NFieldCollectionError);

    for (auto && tup : akantu::zip(Fix::fc.get_pixels(), pixels)) {
      auto && stored_id{std::get<0>(tup)};
      auto && ref_id{std::get<1>(tup)};
      for (int i{0}; i < Fix::spatial_dimension(); ++i) {
        BOOST_CHECK_EQUAL(stored_id[i], ref_id[i]);
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE(initialisation_test_local, LocalFCFixture) {
    constexpr Dim_t NbPixels{6};
    constexpr Dim_t NbQuad{3};
    std::array<Dim_t, NbPixels> indices{0, 12, 46, 548, 6877, 54862};
    BOOST_CHECK_THROW(fc.initialise(), NFieldCollectionError);
    BOOST_CHECK_EQUAL(fc.has_nb_quad(), false);
    for (const auto & index : indices) {
      fc.add_pixel(index);
    }
    fc.set_nb_quad(NbQuad);
    BOOST_CHECK_EQUAL(fc.has_nb_quad(), true);
    BOOST_CHECK_NO_THROW(fc.initialise());
    BOOST_CHECK_THROW(fc.initialise(), NFieldCollectionError);
    BOOST_CHECK_EQUAL(NbPixels * NbQuad, fc.size());
    for (auto && tup : akantu::zip(fc, indices)) {
      auto && stored_id{std::get<0>(tup)};
      auto && ref_id{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(stored_id, ref_id);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
