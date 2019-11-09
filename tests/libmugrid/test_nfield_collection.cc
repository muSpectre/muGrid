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
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include "tests.hh"

#include "libmugrid/nfield_collection_global.hh"
#include "libmugrid/nfield_collection_local.hh"
#include "libmugrid/nfield_map_static.hh"
#include "libmugrid/nfield_typed.hh"
#include "libmugrid/iterators.hh"
#include "libmugrid/ccoord_operations.hh"

#include "test_goodies.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(nfield_collection_test);

  template <Dim_t DimS>
  struct GlobalFCFixture {
    constexpr static Dim_t spatial_dimension() { return DimS; }
    GlobalNFieldCollection fc{DimS, Unknown};
  };

  using GlobalFCFixtures =
      boost::mpl::list<GlobalFCFixture<twoD>, GlobalFCFixture<threeD>>;

  struct LocalFCFixture {
    LocalNFieldCollection fc{Unknown, Unknown};
  };

  template <Dim_t DimS, NFieldCollection::ValidityDomain Domain,
            Dim_t NbQuadPts_>
  struct FC_multi_fixture {
    using FC_t =
        std::conditional_t<(Domain == NFieldCollection::ValidityDomain::Global),
                           GlobalNFieldCollection, LocalNFieldCollection>;
    static constexpr Dim_t SpatialDimension{DimS};
    static constexpr Dim_t NbQuadPts{NbQuadPts_};

    FC_t fc{DimS, NbQuadPts};
    RealNField & t4_field{fc.register_real_field("Tensorfield real o4",
                                                 ipow(SpatialDimension, 4))};
    IntNField & t2_field{fc.register_int_field("Tensorfield integer o2",
                                               ipow(SpatialDimension, 2))};
    UintNField & scalar_field{
        fc.register_uint_field("Scalar unsigned integer", 1)};
    ComplexNField & matrix_field{fc.register_complex_field(
        "Matrixfield Complex sdim × nb_quad", SpatialDimension * NbQuadPts)};
  };
  using mult_collections = boost::mpl::list<
      FC_multi_fixture<twoD, NFieldCollection::ValidityDomain::Global,
                       OneQuadPt>,
      FC_multi_fixture<threeD, NFieldCollection::ValidityDomain::Global,
                       OneQuadPt>,
      FC_multi_fixture<twoD, NFieldCollection::ValidityDomain::Local,
                       OneQuadPt>>;
  using mult_collections_global = boost::mpl::list<
      FC_multi_fixture<twoD, NFieldCollection::ValidityDomain::Global,
                       OneQuadPt>,
      FC_multi_fixture<threeD, NFieldCollection::ValidityDomain::Global,
                       OneQuadPt>>;

  using mult_collections_local = boost::mpl::list<FC_multi_fixture<
      twoD, NFieldCollection::ValidityDomain::Local, OneQuadPt>>;

  BOOST_AUTO_TEST_CASE(NFieldCollection_construction) {
    BOOST_CHECK_NO_THROW(LocalNFieldCollection(Unknown, Unknown));
    using GlobalNFieldCollection2d = GlobalNFieldCollection;
    BOOST_CHECK_NO_THROW(GlobalNFieldCollection2d(twoD, Unknown));
  }

  // the following test only tests members of the NFieldCollection base class,
  // so it is not necessary to test it on both LocalNFieldCollection
  // GlobalNFieldCollection
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(registration_test, Fix, GlobalFCFixtures,
                                   Fix) {
    BOOST_CHECK(Fix::fc.get_domain() ==
                NFieldCollection::ValidityDomain::Global);
    const std::string right_name{"right name"}, wrong_name{"wrong name"};
    Fix::fc.register_real_field(right_name, OneQuadPt);
    const bool should_be_true{Fix::fc.field_exists(right_name)};
    const bool should_be_false{Fix::fc.field_exists(wrong_name)};
    BOOST_CHECK_EQUAL(true, should_be_true);
    BOOST_CHECK_EQUAL(false, should_be_false);
    BOOST_CHECK_EQUAL(Unknown, Fix::fc.get_nb_entries());
    BOOST_CHECK_THROW(Fix::fc.register_real_field(right_name, 24),
                      NFieldCollectionError);
  }

  // the following test only tests members of the NFieldCollection base class,
  // so it is not necessary to test it on both LocalNFieldCollection
  // GlobalNFieldCollection
  BOOST_AUTO_TEST_CASE(local_registration_test) {
    LocalNFieldCollection fc{Unknown, Unknown};
    BOOST_CHECK(fc.get_domain() == NFieldCollection::ValidityDomain::Local);
    const std::string right_name{"right name"}, wrong_name{"wrong name"};
    fc.template register_field<Real>(right_name, OneQuadPt);
    const bool should_be_true{fc.field_exists(right_name)};
    const bool should_be_false{fc.field_exists(wrong_name)};
    BOOST_CHECK_EQUAL(true, should_be_true);
    BOOST_CHECK_EQUAL(false, should_be_false);
    BOOST_CHECK_EQUAL(Unknown, fc.get_nb_entries());
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
    BOOST_CHECK_EQUAL(nb_pixels * NbQuad, Fix::fc.get_nb_entries());

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
    BOOST_CHECK_EQUAL(NbPixels * NbQuad, fc.get_nb_entries());
    for (auto && tup : akantu::zip(fc.get_pixel_indices(), indices)) {
      auto && stored_id{std::get<0>(tup)};
      auto && ref_id{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(stored_id, ref_id);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iteration_test_global, Fix, GlobalFCFixtures,
                                   Fix) {
    Ccoord_t<Fix::spatial_dimension()> nb_grid_pts{};
    Dim_t nb_pixels{1};
    constexpr Dim_t NbQuad{3};
    for (int i{0}; i < Fix::spatial_dimension(); ++i) {
      const auto nb_grid{2 * i + 1};
      nb_grid_pts[i] = nb_grid;
      nb_pixels *= nb_grid;
    }
    Fix::fc.set_nb_quad(NbQuad);
    Fix::fc.initialise(nb_grid_pts);

    for (auto && tup : akantu::enumerate(Fix::fc.get_pixel_indices())) {
      auto && counter{std::get<0>(tup)};
      auto && pixel_index{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(counter, pixel_index);
    }
    for (auto && tup : akantu::enumerate(Fix::fc.get_quad_pt_indices())) {
      auto && counter{std::get<0>(tup)};
      auto && quad_pt_index{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(counter, quad_pt_index);
    }
  }

  BOOST_FIXTURE_TEST_CASE(iteration_test_local, LocalFCFixture) {
    constexpr Dim_t NbPixels{2};
    constexpr Dim_t NbQuad{3};
    std::array<Dim_t, NbPixels> pixel_indices{0, 12};
    std::array<Dim_t, NbPixels * NbQuad> quad_pt_indices{};
    size_t counter{};
    for (const auto & pixel_index : pixel_indices) {
      for (Dim_t i{0}; i < NbQuad; ++i, ++counter) {
        quad_pt_indices[counter] = pixel_index * NbQuad + i;
      }
    }
    for (const auto & index : pixel_indices) {
      fc.add_pixel(index);
    }
    fc.set_nb_quad(NbQuad);

    for (auto && tup : akantu::zip(fc.get_pixel_indices(), pixel_indices)) {
      auto && stored_id{std::get<0>(tup)};
      auto && ref_id{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(stored_id, ref_id);
    }

    for (auto && tup : akantu::zip(fc.get_quad_pt_indices(), quad_pt_indices)) {
      auto && stored_id{std::get<0>(tup)};
      auto && ref_id{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(stored_id, ref_id);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(global_clone, Fix, GlobalFCFixtures, Fix) {
    Ccoord_t<Fix::spatial_dimension()> nb_grid_pts{};
    Dim_t nb_pixels{1};
    constexpr Dim_t NbQuad{3};
    for (int i{0}; i < Fix::spatial_dimension(); ++i) {
      const auto nb_grid{2 * i + 1};
      nb_grid_pts[i] = nb_grid;
      nb_pixels *= nb_grid;
    }
    Fix::fc.set_nb_quad(NbQuad);
    Fix::fc.initialise(nb_grid_pts);

    auto fc2{Fix::fc.get_empty_clone()};
    BOOST_CHECK_EQUAL(fc2.get_nb_quad(), Fix::fc.get_nb_quad());
    BOOST_CHECK_EQUAL(fc2.get_nb_entries(), Fix::fc.get_nb_entries());
    BOOST_CHECK_EQUAL(fc2.get_spatial_dim(), Fix::fc.get_spatial_dim());
    BOOST_CHECK_EQUAL(fc2.get_nb_pixels(), Fix::fc.get_nb_pixels());
  }

  BOOST_FIXTURE_TEST_CASE(local_clone, LocalFCFixture) {
    constexpr Dim_t NbPixels{2};
    constexpr Dim_t NbQuad{3};
    std::array<Dim_t, NbPixels> pixel_indices{0, 12};

    for (const auto & index : pixel_indices) {
      fc.add_pixel(index);
    }
    fc.set_nb_quad(NbQuad);

    auto fc2{fc.get_empty_clone()};
    BOOST_CHECK_EQUAL(fc2.get_nb_quad(), fc.get_nb_quad());
    BOOST_CHECK_EQUAL(fc2.get_nb_entries(), fc.get_nb_entries());
    BOOST_CHECK_EQUAL(fc2.get_spatial_dim(), fc.get_spatial_dim());
    BOOST_CHECK_EQUAL(fc2.get_nb_pixels(), fc.get_nb_pixels());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(multi_field_test, F, mult_collections, F) {
    // possible maptypes for Real T4 fields

    using T_TFM1_t = T4NFieldMap<Real, Mapping::Const, F::SpatialDimension>;
    using T_TFM2_t =
        T2NFieldMap<Real, Mapping::Const, ipow(F::SpatialDimension, 2)>;
    using T4_Map_t =
        MatrixNFieldMap<Real, Mapping::Mut, ipow(F::SpatialDimension, 4),
                        F::NbQuadPts, Iteration::Pixel>;

    // impossible maptypes for Real tensor fields
    using T_SFM_t = ScalarNFieldMap<Real, Mapping::Mut>;
    using T_MFM_t =
        MatrixNFieldMap<Real, Mapping::Mut, ipow(F::SpatialDimension, 4),
                        F::NbQuadPts + 1, Iteration::Pixel>;
    using T_MFMw1_t =
        MatrixNFieldMap<Int, Mapping::Mut, ipow(F::SpatialDimension, 4),
                        F::NbQuadPts, Iteration::Pixel>;
    const std::string T_name{"Tensorfield real o4"};
    const std::string T_name_w{"TensorField Real o4 wrongname"};

    BOOST_CHECK_THROW(T_SFM_t(F::fc.get_field(T_name)), NFieldMapError);
    BOOST_CHECK_NO_THROW(T_TFM1_t(F::fc.get_field(T_name)));
    BOOST_CHECK_NO_THROW(T_TFM2_t(F::fc.get_field(T_name)));
    BOOST_CHECK_NO_THROW(T4_Map_t(F::fc.get_field(T_name)));
    BOOST_CHECK_THROW(T4_Map_t(F::fc.get_field(T_name_w)),
                      NFieldCollectionError);
    BOOST_CHECK_THROW(T_MFM_t(F::fc.get_field(T_name)), NFieldMapError);
    BOOST_CHECK_THROW(T_MFMw1_t(F::fc.get_field(T_name)), NFieldError);
    BOOST_CHECK_THROW(T_SFM_t(F::fc.get_field(T_name_w)),
                      NFieldCollectionError);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca_with_push_back, F,
                                   mult_collections_local, F) {
    constexpr int nb_pix{7};
    testGoodies::RandRange<Int> rng{};
    using stype = Eigen::Array<Real, ipow(F::SpatialDimension, 4), 1>;
    auto & field{RealNField::safe_cast(F::fc.get_field("Tensorfield real o4"),
                                       ipow(F::SpatialDimension, 4))};
    field.push_back(stype());
    for (int i = 0; i < nb_pix; ++i) {
      F::fc.add_pixel(rng.randval(0, nb_pix));
    }

    BOOST_CHECK_THROW(F::fc.initialise(), NFieldCollectionError);
    for (int i = 0; i < nb_pix - 1; ++i) {
      field.push_back(stype());
    }
    BOOST_CHECK_NO_THROW(F::fc.initialise());
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
