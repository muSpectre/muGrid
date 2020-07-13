/**
 * @file   test_mapped_fields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Sep 2019
 *
 * @brief  Tests for the mapped field classes
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
#include "field_test_fixtures.hh"

#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/mapped_field.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(mapped_fields);

  struct InitialiserBase {
    constexpr static Dim_t DimS{twoD};
    constexpr static Dim_t NbRow{2}, NbCol{3};
    constexpr static Dim_t NbQuad{2};
    constexpr static Dim_t NbNode{4};
    GlobalFieldCollection fc{DimS};
    static std::string division_tag() { return "tag"; }
    InitialiserBase() {
      this->fc.initialise(Ccoord_t<twoD>{2, 3});
      this->fc.set_nb_sub_pts(division_tag(), NbQuad);
    }
  };
  constexpr Dim_t InitialiserBase::DimS;
  constexpr Dim_t InitialiserBase::NbRow;
  constexpr Dim_t InitialiserBase::NbCol;
  constexpr Dim_t InitialiserBase::NbQuad;
  constexpr Dim_t InitialiserBase::NbNode;

  struct MappedFieldFixture : public InitialiserBase {
    MappedMatrixField<Real, Mapping::Mut, NbRow, NbCol, IterUnit::SubPt>
        mapped_matrix;
    MappedArrayField<Real, Mapping::Mut, NbRow, NbCol, IterUnit::SubPt>
        mapped_array;
    MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt> mapped_scalar;
    MappedT2Field<Real, Mapping::Mut, DimS, IterUnit::SubPt> mapped_t2;
    MappedT4Field<Real, Mapping::Mut, DimS, IterUnit::SubPt> mapped_t4;


    MappedFieldFixture()
        : InitialiserBase{}, mapped_matrix{"matrix", this->fc, division_tag()},
          mapped_array{"array", this->fc, division_tag()},
          mapped_scalar{"scalar", this->fc, division_tag()},
          mapped_t2{"t2", this->fc, division_tag()}, mapped_t4{
                                                         "t4", this->fc,
                                                         division_tag()} {};
  };

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(access_and_iteration_test, MappedFieldFixture) {
    for (auto && iterate : this->mapped_t2) {
      iterate.setRandom();
    }
    for (auto && iterate : this->mapped_matrix) {
      iterate.setRandom();
    }
    this->mapped_array.get_field().eigen_sub_pt() =
        this->mapped_matrix.get_field().eigen_sub_pt();
    for (auto && tup : akantu::zip(this->mapped_matrix, this->mapped_array)) {
      const auto & matrix{std::get<0>(tup)};
      const auto & array{std::get<1>(tup)};
      BOOST_CHECK_EQUAL((matrix - array.matrix()).norm(), 0);
    }

    BOOST_CHECK_EQUAL(
        (this->mapped_array[4].matrix() - this->mapped_matrix[4]).norm(), 0);
  }
  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(DynamicMappedField) {
    constexpr Dim_t NbQuad{3};
    constexpr Dim_t NbRow{4};
    constexpr Dim_t NbCol{5};
    const std::string division_tag{"tag"};
    GlobalFieldCollection collection{twoD};
    collection.set_nb_sub_pts(division_tag, NbQuad);
    collection.initialise({2, 2});
    using Mapped_t = MappedField<FieldMap<Real, Mapping::Mut>>;
    Mapped_t mapped_field{"name", NbRow, NbCol, IterUnit::SubPt,
                          collection, division_tag};

    BOOST_CHECK_THROW(Mapped_t("name", NbRow, NbCol, IterUnit::Pixel,
                               collection, division_tag),
                      FieldCollectionError);

    MatrixFieldMap<Real, Mapping::Const, NbRow, NbCol, IterUnit::SubPt>
        static_map(mapped_field.get_field());

    for (auto && tup : akantu::zip(mapped_field.get_map(), static_map)) {
      auto & dyn{std::get<0>(tup)};
      auto & stat{std::get<1>(tup)};
      dyn.setRandom();
      const Real error{(dyn - stat).norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }
  }

  using ValidityDomain = FieldCollection::ValidityDomain;

  /* ---------------------------------------------------------------------- */
  template <ValidityDomain Validity_>
  struct FieldFixture {
    constexpr static ValidityDomain Validity{Validity_};
    constexpr static Dim_t Order{secondOrder};
    constexpr static Dim_t SDim{twoD};
    constexpr static Dim_t MDim{threeD};
    constexpr static Dim_t NbComponents{ipow(MDim, Order)};
    static const std::string subdivision_tag() { return "quad"; }
    using FieldColl_t =
        std::conditional_t<Validity == ValidityDomain::Global,
                           GlobalFieldCollection, LocalFieldCollection>;
    using TField_t =
        MappedT2Field<Real, Mapping::Mut, MDim, IterUnit::SubPt>;
    using MField_t =
        MappedMatrixField<Real, Mapping::Mut, SDim, MDim, IterUnit::SubPt>;
    using DField_t = RealField;

    FieldFixture()
        : tensor_field{"TensorField", this->fc, subdivision_tag()},
          matrix_field{"MatrixField", this->fc, subdivision_tag()},
          dynamic_field1{this->fc.register_real_field(
              "Dynamically sized field with correct number of"
              " components",
              NbComponents, subdivision_tag())},
          dynamic_field2{this->fc.register_real_field(
              "Dynamically sized field with incorrect number"
              " of components",
              NbComponents + 1, subdivision_tag())} {
      fc.set_nb_sub_pts(subdivision_tag(), OneQuadPt);
    }
    ~FieldFixture() = default;

    FieldColl_t fc{SDim};
    TField_t tensor_field;
    MField_t matrix_field;
    DField_t & dynamic_field1;
    DField_t & dynamic_field2;
  };

  template <ValidityDomain Validity>
  constexpr Dim_t FieldFixture<Validity>::NbComponents;
  template <ValidityDomain Validity>
  constexpr Dim_t FieldFixture<Validity>::SDim;

  using field_fixtures = boost::mpl::list<FieldFixture<ValidityDomain::Local>,
                                          FieldFixture<ValidityDomain::Global>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(size_check_global,
                          FieldFixture<ValidityDomain::Global>) {
    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), Unknown);

    // check that returned size is correct
    Index_t len{2};
    auto sizes{CcoordOps::get_cube<SDim>(len)};
    fc.initialise(sizes, {});

    auto nb_pixels{CcoordOps::get_size(sizes)};
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(),
                      nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), nb_pixels);

    constexpr Index_t pad_size{3};
    tensor_field.get_field().set_pad_size(pad_size);
    dynamic_field1.set_pad_size(pad_size);
    dynamic_field2.set_pad_size(pad_size);

    // check that setting pad size won't change logical size
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(),
                      nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), nb_pixels);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(size_check_local,
                          FieldFixture<ValidityDomain::Local>) {
    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), Unknown);

    // check that returned size is correct
    Index_t nb_pixels{3};

    Eigen::Array<Real, NbComponents, 1> new_elem;
    Eigen::Array<Real, 1, NbComponents + 1> wrong_elem;
    for (Index_t i{0}; i < NbComponents; ++i) {
      new_elem(i) = i;
      wrong_elem(i) = .1 * i;
    }

    for (Index_t i{0}; i < nb_pixels; ++i) {
      tensor_field.get_field().push_back(new_elem);
      dynamic_field1.push_back(new_elem);
      BOOST_CHECK_THROW(dynamic_field1.push_back(wrong_elem), FieldError);
      BOOST_CHECK_THROW(dynamic_field2.push_back(new_elem), FieldError);
    }
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), Unknown);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), Unknown);

    for (Index_t i{0}; i < nb_pixels; ++i) {
      fc.add_pixel(i);
    }

    fc.initialise();

    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), nb_pixels);

    BOOST_CHECK_EQUAL(tensor_field.get_field().get_pad_size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field1.get_pad_size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field2.get_pad_size(), 0);

    constexpr Index_t pad_size{3};
    tensor_field.get_field().set_pad_size(pad_size);
    dynamic_field1.set_pad_size(pad_size);
    dynamic_field2.set_pad_size(pad_size);

    BOOST_CHECK_EQUAL(tensor_field.get_field().get_pad_size(), pad_size);
    BOOST_CHECK_EQUAL(dynamic_field1.get_pad_size(), pad_size);
    BOOST_CHECK_EQUAL(dynamic_field2.get_pad_size(), pad_size);

    // check that setting pad size won't change logical size
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.get_nb_entries(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.get_nb_entries(), nb_pixels);

    // check that the buffer size is correct
    BOOST_CHECK_EQUAL(tensor_field.get_field().get_buffer_size(),
                      nb_pixels * tensor_field.get_field().get_nb_components() +
                          pad_size);
    BOOST_CHECK_EQUAL(dynamic_field1.get_buffer_size(),
                      nb_pixels * dynamic_field1.get_nb_components() +
                          pad_size);
    BOOST_CHECK_EQUAL(dynamic_field2.get_buffer_size(),
                      nb_pixels * dynamic_field2.get_nb_components() +
                          pad_size);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(mapped_fields_subdivision, SubDivisionFixture) {
    MappedArrayField<Real, Mapping::Mut, NbComponent, 1, IterUnit::Pixel>
      pixel{"pixel", this->fc, PixelTag};
    MappedArrayField<Real, Mapping::Mut, NbComponent, 1, IterUnit::SubPt>
        quad_pt{"quad_pt", this->fc, sub_division_tag()};
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
