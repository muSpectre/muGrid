/**
 * @file   test_field_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Aug 2019
 *
 * @brief  tests for field maps
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

#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/field_map_static.hh"
#include "libmugrid/iterators.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(field_maps);

  struct BaseFixture {
    constexpr static Dim_t NbQuadPts() { return 2; }
    constexpr static Dim_t Dim() { return threeD; }
  };

  struct GlobalFieldCollectionFixture : public BaseFixture {
    GlobalFieldCollectionFixture()
        : fc{BaseFixture::Dim(), BaseFixture::NbQuadPts()} {
      Ccoord_t<BaseFixture::Dim()> nb_grid_pts{2, 2, 3};
      this->fc.initialise(nb_grid_pts);
    }
    GlobalFieldCollection fc;
    constexpr static Dim_t size{12};
  };

  struct LocalFieldCollectionFixture : public BaseFixture {
    LocalFieldCollectionFixture()
        : fc{BaseFixture::Dim(), BaseFixture::NbQuadPts()} {
      this->fc.add_pixel(0);
      this->fc.add_pixel(11);
      this->fc.add_pixel(102);
      this->fc.initialise();
    }
    LocalFieldCollection fc;
    constexpr static size_t size{3};
  };

  template <typename T, class CollectionFixture>
  struct FieldMapFixture : public CollectionFixture {
    using type = T;
    FieldMapFixture()
        : scalar_field{this->fc.template register_field<T>("scalar_field", 1)},
          vector_field{this->fc.template register_field<T>("vector_field",
                                                           BaseFixture::Dim())},
          matrix_field{this->fc.template register_field<T>(
              "matrix_field", BaseFixture::Dim() * BaseFixture::Dim())},
          T4_field{this->fc.template register_field<T>(
              "tensor4_field", ipow(BaseFixture::Dim(), 4))},
          scalar_quad{scalar_field, Iteration::QuadPt},
          scalar_pixel{scalar_field, Iteration::Pixel},
          vector_quad{vector_field, Iteration::QuadPt},
          vector_pixel{vector_field, Iteration::Pixel},
          matrix_quad{matrix_field, BaseFixture::Dim(), Iteration::QuadPt},
          matrix_pixel{matrix_field, BaseFixture::Dim(), Iteration::Pixel} {}
    TypedField<T> & scalar_field;
    TypedField<T> & vector_field;
    TypedField<T> & matrix_field;
    TypedField<T> & T4_field;

    FieldMap<T, Mapping::Mut> scalar_quad;
    FieldMap<T, Mapping::Mut> scalar_pixel;
    FieldMap<T, Mapping::Mut> vector_quad;
    FieldMap<T, Mapping::Mut> vector_pixel;
    FieldMap<T, Mapping::Mut> matrix_quad;
    FieldMap<T, Mapping::Mut> matrix_pixel;
  };

  using Maps = boost::mpl::list<
      FieldMapFixture<Real, LocalFieldCollectionFixture>,
      FieldMapFixture<Complex, LocalFieldCollectionFixture>,
      FieldMapFixture<Real, GlobalFieldCollectionFixture>,
      FieldMapFixture<Complex, GlobalFieldCollectionFixture>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(construction_test, Fix, Maps, Fix) {
    typename FieldMap<typename Fix::type,
                       Mapping::Mut>::template Iterator<Mapping::Mut>
        beg{Fix::scalar_quad.begin()};
    BOOST_CHECK_EQUAL((*beg).size(), 1);
    BOOST_CHECK_EQUAL((*Fix::scalar_pixel.begin()).size(), Fix::NbQuadPts());
    // check also const version
    const auto & const_scalar_pixel{Fix::scalar_pixel};
    BOOST_CHECK_EQUAL((*const_scalar_pixel.begin()).size(), Fix::NbQuadPts());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iteration_test, Fix, Maps, Fix) {
    for (auto && tup : akantu::enumerate(Fix::scalar_pixel)) {
      const auto & i{std::get<0>(tup)};
      auto iterate{std::get<1>(tup)};
      iterate(0, 0) = i;
    }
    for (auto && tup : akantu::enumerate(Fix::scalar_pixel)) {
      const auto & i{std::get<0>(tup)};
      auto & iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(iterate.norm(), i);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_size_test, Fix, Maps, Fix) {
    using StaticMap_t = MatrixFieldMap<typename Fix::type, Mapping::Const,
                                        Fix::Dim(), Fix::Dim()>;
    StaticMap_t static_map{Fix::matrix_field};
    for (auto && iterate : Fix::matrix_pixel) {
      iterate.setRandom();
    }
    for (auto && tup : akantu::zip(Fix::matrix_quad, static_map)) {
      const auto & dynamic_iterate{std::get<0>(tup)};
      const auto & static_iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL((dynamic_iterate - static_iterate).norm(), 0);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_size_test_pixels, Fix, Maps, Fix) {
    using StaticMatrixMap_t =
        MatrixFieldMap<typename Fix::type, Mapping::Const, Fix::Dim(),
                        Fix::Dim() * Fix::NbQuadPts(), Iteration::Pixel>;
    StaticMatrixMap_t static_map{Fix::matrix_field};
    for (auto && iterate : Fix::matrix_pixel) {
      iterate.setRandom();
    }
    for (auto && tup : akantu::zip(Fix::matrix_pixel, static_map)) {
      const auto & dynamic_iterate{std::get<0>(tup)};
      const auto & static_iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL((dynamic_iterate - static_iterate).norm(), 0);
    }

    using ScalarMap_t = ScalarFieldMap<typename Fix::type, Mapping::Mut>;
    ScalarMap_t scalar_map{Fix::scalar_field};

    for (auto && tup : akantu::zip(Fix::scalar_quad, scalar_map)) {
      const auto & dynamic_iterate{std::get<0>(tup)};
      auto & static_iterate{std::get<1>(tup)};
      static_iterate += 1;
      BOOST_CHECK_EQUAL(std::abs(dynamic_iterate(0, 0) - static_iterate), 0);
    }

    // testing both operators -> and *
    for (auto it{static_map.begin()}; it != static_map.end(); ++it) {
      BOOST_CHECK_EQUAL((*it).norm(), it->norm());
    }

    // testing array map and t2 map
    using StaticArrayMap_t =
        ArrayFieldMap<typename Fix::type, Mapping::Const, Fix::Dim(),
                       Fix::Dim(), Iteration::QuadPt>;
    StaticArrayMap_t array_map{Fix::matrix_field};
    using T2Map_t = T2FieldMap<typename Fix::type, Mapping::Const, Fix::Dim()>;
    T2Map_t t2_map{Fix::matrix_field};

    for (auto && tup : akantu::zip(array_map, t2_map)) {
      const auto & array_iterate{std::get<0>(tup)};
      const auto & t2_iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL((array_iterate.matrix() - t2_iterate).norm(), 0);
    }
    // testing t4 map
    using StaticMatrix4Map_t =
        MatrixFieldMap<typename Fix::type, Mapping::Const,
                        Fix::Dim() * Fix::Dim(), Fix::Dim() * Fix::Dim(),
                        Iteration::QuadPt>;
    StaticMatrix4Map_t matrix4_map{Fix::T4_field};

    using T4Map_t = T4FieldMap<typename Fix::type, Mapping::Mut, Fix::Dim()>;
    T4Map_t t4_map{Fix::T4_field};
    Fix::T4_field.eigen_vec().setRandom();

    for (auto && tup : akantu::zip(matrix4_map, t4_map)) {
      const auto & matrix4_iterate{std::get<0>(tup)};
      const auto & t4_iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL((matrix4_iterate.matrix() - t4_iterate).norm(), 0);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(enumeration_test, Fix, Maps, Fix) {
    for (auto && tup :
         akantu::zip(Fix::fc.get_pixel_indices(), Fix::scalar_pixel)) {
      const auto & i{std::get<0>(tup)};
      auto iterate{std::get<1>(tup)};
      iterate(0, 0) = i;
    }
    for (auto && tup : Fix::scalar_pixel.enumerate_pixel_indices_fast()) {
      const auto & i{std::get<0>(tup)};
      auto & iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(iterate.norm(), i);
    }

    BOOST_CHECK_THROW(Fix::scalar_quad.enumerate_pixel_indices_fast(),
                      FieldMapError);

    for (auto && tup :
         akantu::zip(Fix::fc.get_quad_pt_indices(), Fix::scalar_quad)) {
      const auto & i{std::get<0>(tup)};
      auto iterate{std::get<1>(tup)};
      iterate(0, 0) = i;
    }

    for (auto && tup : Fix::scalar_quad.enumerate_indices()) {
      const auto & i{std::get<0>(tup)};
      auto & iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(iterate.norm(), i);
    }

    for (auto && val : Fix::scalar_quad.enumerate_indices()) {
      static_assert(std::remove_reference_t<decltype(std::get<1>(
                            val))>::SizeAtCompileTime == Eigen::Dynamic,
                    "Should be testing the dynamic maps");
    }
    for (auto && val : Fix::scalar_pixel.enumerate_indices()) {
      static_assert(std::remove_reference_t<decltype(std::get<1>(
                            val))>::SizeAtCompileTime == Eigen::Dynamic,
                    "Should be testing the dynamic maps");
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_enumeration_test, Fix, Maps, Fix) {
    ScalarFieldMap<typename Fix::type, Mapping::Const> static_scalar_quad{
        Fix::scalar_field};
    MatrixFieldMap<typename Fix::type, Mapping::Const, Fix::NbQuadPts(), 1,
                    Iteration::Pixel>
        static_scalar_pixel{Fix::scalar_field};
    for (auto && tup :
         akantu::zip(Fix::fc.get_pixel_indices(), Fix::scalar_pixel)) {
      const auto & i{std::get<0>(tup)};
      auto iterate{std::get<1>(tup)};
      iterate(0, 0) = i;
    }
    for (auto && tup : static_scalar_pixel.enumerate_indices()) {
      const auto & i{std::get<0>(tup)};
      auto & iterate{std::get<1>(tup)};
      BOOST_CHECK_EQUAL(iterate.norm(), i);
    }

    for (auto && tup :
         akantu::zip(Fix::fc.get_quad_pt_indices(), Fix::scalar_quad)) {
      const auto & i{std::get<0>(tup)};
      auto iterate{std::get<1>(tup)};
      iterate(0, 0) = i;
    }

    for (auto && tup : static_scalar_quad.enumerate_indices()) {
      const auto & i{std::get<0>(tup)};
      auto & iterate{std::get<1>(tup)};
      // the std::remove_reference_t<decltype(... is because of the Complex case
      BOOST_CHECK_EQUAL(iterate, std::remove_reference_t<decltype(iterate)>(i));
    }

    for (auto && val : static_scalar_quad.enumerate_indices()) {
      using Iterate_t = std::remove_reference_t<decltype(std::get<1>(val))>;

      static_assert(std::is_same<Iterate_t, const typename Fix::type>::value,
                    "Should be a scalar");
    }
    for (auto && val : static_scalar_pixel.enumerate_indices()) {
      static_assert(std::remove_reference_t<decltype(std::get<1>(
                            val))>::SizeAtCompileTime != Eigen::Dynamic,
                    "Should be testing the dynamic maps");
    }
  }
  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
