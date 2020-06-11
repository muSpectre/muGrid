/**
 * @file   test_field.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Testing fields
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
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/field_collection_global.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(field_test);

  struct LocalFieldBasicFixture {
    LocalFieldCollection fc{Unknown};
    static std::string SubDivision() { return "quad"; }
  };

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(simple_creation) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t MDim{twoD};
    using FC_t = GlobalFieldCollection;
    FC_t fc{SDim};
    fc.set_nb_sub_pts("quad", OneQuadPt);

    auto & field{fc.register_real_field("TensorField 1", MDim * MDim, "quad")};

    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(field.size(), 0);
    Index_t len{2};
    fc.initialise(CcoordOps::get_cube<SDim>(len), {});
    // check that returned size is correct
    BOOST_CHECK_EQUAL(field.size(), ipow(len, SDim));
    // check that setting pad size won't change logical size
    field.set_pad_size(24);
    BOOST_CHECK_EQUAL(field.size(), ipow(len, SDim));
  }

  BOOST_AUTO_TEST_CASE(sub_divisions_with_known) {
    static const std::string quad{"quad"};
    static const std::string nodal{"nodal"};
    constexpr Index_t SDim{twoD};
    constexpr Index_t MDim{twoD};
    constexpr Index_t NbQuad{OneQuadPt};
    constexpr Index_t NbNode{OneNode};
    constexpr Index_t NbComponent{MDim};
    using FC_t = GlobalFieldCollection;
    FC_t fc{SDim};
    fc.set_nb_sub_pts(quad, NbQuad);
    fc.set_nb_sub_pts(nodal, NbNode);

    auto & pixel_field{fc.register_real_field("Pixel", NbComponent, PixelTag)};
    auto & nodal_field{fc.register_real_field("Nodal", NbComponent, nodal)};
    auto & quad_field{fc.register_real_field("Quad", NbComponent, quad)};

    Index_t len{2};
    Index_t nb_pix{ipow(len, SDim)};
    fc.initialise(CcoordOps::get_cube<SDim>(len), {});

    BOOST_CHECK_EQUAL(pixel_field.size(), nb_pix);
    BOOST_CHECK_EQUAL(nodal_field.size(), nb_pix * NbNode);
    BOOST_CHECK_EQUAL(quad_field.size(), nb_pix * NbQuad);
  }

  BOOST_AUTO_TEST_CASE(TypedField_local_filling) {
    static const std::string pixel{"pixel"};
    static const std::string quad{"quad"};
    static const std::string nodal{"nodal"};
    LocalFieldCollection fc{Unknown};
    constexpr Dim_t NbComponents{3}, NbQuadPts{4};
    auto & scalar_field{fc.register_field<Real>("scalar_field", 1, quad)};
    auto & vector_field{
        fc.register_field<Real>("vector_field", NbComponents, quad)};
    const bool is_same_type{scalar_field.get_stored_typeid() == typeid(Real)};
    BOOST_CHECK(is_same_type);

    BOOST_CHECK_THROW(scalar_field.push_back(Real{3}), FieldError);
    Eigen::Matrix<Real, 1, 1> scalar_mat{};
    scalar_mat.setZero();
    BOOST_CHECK_THROW(scalar_field.push_back(scalar_mat.array()), FieldError);

    fc.set_nb_sub_pts(quad, NbQuadPts);
    scalar_field.push_back(Real{3});
    scalar_field.push_back(scalar_mat.array());

    BOOST_CHECK_THROW(vector_field.push_back(Real{3}), FieldError);

    Eigen::Matrix<Real, NbComponents, 1> vector_mat{};
    vector_mat.setZero();
    vector_field.push_back(vector_mat.array());
    vector_field.push_back(vector_mat.Ones().array());

    BOOST_CHECK_THROW(fc.initialise(), FieldCollectionError);
    BOOST_CHECK_EQUAL(vector_field.size(), scalar_field.size());
    BOOST_CHECK_THROW(fc.initialise(), FieldCollectionError);
    fc.add_pixel(0);
    BOOST_CHECK_THROW(fc.initialise(), FieldCollectionError);
    fc.add_pixel(1);
    BOOST_CHECK_NO_THROW(fc.initialise());
  }

  BOOST_AUTO_TEST_CASE(TypedField_globel_not_filling) {
    GlobalFieldCollection fc{twoD};
    const std::string quad{"quad"};
    constexpr Dim_t NbComponents{3};
    auto & scalar_field{fc.register_field<Real>("scalar_field", 1, quad)};
    auto & vector_field{
        fc.register_field<Real>("vector_field", NbComponents, quad)};

    BOOST_CHECK_THROW(scalar_field.push_back(3.7), FieldError);
    Eigen::Matrix<Real, NbComponents, 1> vector_mat{};
    vector_mat.setZero();
    BOOST_CHECK_THROW(vector_field.push_back(vector_mat.array()), FieldError);
  }

  BOOST_FIXTURE_TEST_CASE(set_zero, LocalFieldBasicFixture) {
    constexpr Dim_t NbSubPts{3}, NbComponents{4};
    fc.set_nb_sub_pts(SubDivision(), NbSubPts);
    auto & scalar_field{
        fc.register_field<Real>("scalar_field", 1, SubDivision())};
    auto & vector_field{
        fc.register_field<Real>("vector_field", NbComponents, SubDivision())};
    scalar_field.push_back(1);
    vector_field.push_back(Eigen::Array<Real, NbComponents, 1>::Ones());
    fc.add_pixel(24);
    fc.initialise();

    const Eigen::MatrixXd scalar_copy_before{scalar_field.eigen_vec()};
    const Eigen::MatrixXd vector_copy_before{vector_field.eigen_vec()};
    scalar_field.set_zero();
    vector_field.set_zero();
    const Eigen::MatrixXd scalar_copy_after{scalar_field.eigen_vec()};
    const Eigen::MatrixXd vector_copy_after{vector_field.eigen_vec()};

    auto std{[](const auto & mat) -> Real {
      return std::sqrt((mat.array() - mat.mean()).square().sum() /
                       (mat.size()));
    }};

    Real error{scalar_copy_before.mean() - 1.};
    BOOST_CHECK_EQUAL(error, 0.);
    error = std(scalar_copy_before);
    BOOST_CHECK_EQUAL(error, 0.);

    error = vector_copy_before.mean() - 1.;
    BOOST_CHECK_EQUAL(error, 0.);
    error = std(vector_copy_before);
    BOOST_CHECK_EQUAL(error, 0.);

    error = scalar_copy_after.norm();
    BOOST_CHECK_EQUAL(error, 0.);
    error = vector_copy_after.norm();
    BOOST_CHECK_EQUAL(error, 0.);
  }

  BOOST_FIXTURE_TEST_CASE(eigen_maps, LocalFieldBasicFixture) {
    constexpr Dim_t NbSubPts{2}, NbComponents{3};
    fc.set_nb_sub_pts(SubDivision(), NbSubPts);
    auto & vector_field{fc.register_field<Real>("vector_field", NbComponents,
                                                SubDivision())};
    const auto & cvector_field{vector_field};
    fc.add_pixel(0);
    fc.add_pixel(1);
    fc.add_pixel(2);

    fc.initialise();

    using Map_t = TypedFieldBase<Real>::Eigen_map;
    using CMap_t = TypedFieldBase<Real>::Eigen_cmap;
    Map_t vector{vector_field.eigen_vec()};
    CMap_t cvector{cvector_field.eigen_vec()};
    BOOST_CHECK_EQUAL(vector.size(),
                      vector_field.get_nb_entries() * NbComponents);
    for (int i{0}; i < vector.size(); ++i) {
      vector(i) = i;
      BOOST_CHECK_EQUAL(vector(i), cvector(i));
    }

    Map_t quad_map{vector_field.eigen_sub_pt()};
    CMap_t quad_cmap{cvector_field.eigen_sub_pt()};
    BOOST_CHECK_EQUAL(quad_cmap.cols(), vector_field.get_nb_entries());
    BOOST_CHECK_EQUAL(quad_cmap.rows(), NbComponents);
    for (int i{0}; i < quad_cmap.rows(); ++i) {
      for (int j{0}; j < quad_cmap.cols(); ++j) {
        BOOST_CHECK_EQUAL(quad_map(i, j), quad_cmap(i, j));
        BOOST_CHECK_EQUAL(quad_map(i, j), i + NbComponents * j);
      }
    }

    Map_t pixel_map{vector_field.eigen_pixel()};
    CMap_t pixel_cmap{cvector_field.eigen_pixel()};
    BOOST_CHECK_EQUAL(pixel_cmap.cols(),
                      cvector_field.get_nb_entries() / NbSubPts);
    BOOST_CHECK_EQUAL(pixel_cmap.rows(), NbComponents * NbSubPts);
    for (int i{0}; i < pixel_cmap.rows(); ++i) {
      for (int j{0}; j < pixel_cmap.cols(); ++j) {
        BOOST_CHECK_EQUAL(pixel_map(i, j), pixel_cmap(i, j));
        BOOST_CHECK_EQUAL(pixel_map(i, j), i + NbComponents * NbSubPts * j);
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
