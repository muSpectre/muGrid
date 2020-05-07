/**
 * @file   field_test_fixtures.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   17 May 2020
 *
 * @brief  Useful fixtures to perform tests on pre-created fields, field_maps,
 *         and field_collections
 *
 * Copyright © 2020 Till Junge
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

#ifndef TESTS_LIBMUGRID_FIELD_TEST_FIXTURES_HH_
#define TESTS_LIBMUGRID_FIELD_TEST_FIXTURES_HH_

#include "tests.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/field_map_static.hh"

namespace muGrid {
  struct BaseFixture {
    constexpr static Dim_t NbQuadPts{2};
    constexpr static Dim_t NbNodalPts{3};
    constexpr static Dim_t Dim{threeD};
  };

  struct GlobalFieldCollectionFixture : public BaseFixture {
    GlobalFieldCollectionFixture()
        : fc{BaseFixture::Dim, BaseFixture::NbQuadPts,
             BaseFixture::NbNodalPts} {
      Ccoord_t<BaseFixture::Dim> nb_grid_pts{2, 2, 3};
      this->fc.initialise(nb_grid_pts);
    }
    GlobalFieldCollection fc;
    constexpr static Dim_t size{12};
  };

  struct LocalFieldCollectionFixture : public BaseFixture {
    LocalFieldCollectionFixture()
        : fc{BaseFixture::Dim, BaseFixture::NbQuadPts,
             BaseFixture::NbNodalPts} {
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
        : CollectionFixture{}, scalar_field{this->fc.template register_field<T>(
                                   "scalar_field", 1, PixelSubDiv::QuadPt)},
          vector_field{this->fc.template register_field<T>(
              "vector_field", BaseFixture::Dim, PixelSubDiv::QuadPt)},
          matrix_field{this->fc.template register_field<T>(
              "matrix_field", BaseFixture::Dim * BaseFixture::Dim,
              PixelSubDiv::QuadPt)},
          T4_field{this->fc.template register_field<T>(
              "tensor4_field", ipow(BaseFixture::Dim, 4), PixelSubDiv::QuadPt)},
          scalar_quad{scalar_field, PixelSubDiv::QuadPt},
          scalar_pixel{scalar_field, PixelSubDiv::Pixel},
          vector_quad{vector_field, PixelSubDiv::QuadPt},
          vector_pixel{vector_field, PixelSubDiv::Pixel},
          matrix_quad{matrix_field, BaseFixture::Dim, PixelSubDiv::QuadPt},
          matrix_pixel{matrix_field, BaseFixture::Dim, PixelSubDiv::Pixel} {}
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

  /* ---------------------------------------------------------------------- */
  struct SubDivisionFixture : public GlobalFieldCollectionFixture {
    constexpr static Dim_t NbFreeSubDiv{7};
    constexpr static Dim_t NbComponent{2};
    SubDivisionFixture()
        : GlobalFieldCollectionFixture{},
          pixel_field{this->fc.register_real_field("pixel_field", NbComponent,
                                                   PixelSubDiv::Pixel)},
          quad_pt_field{this->fc.register_real_field(
              "quad_pt_field", NbComponent, PixelSubDiv::QuadPt)},
          nodal_pt_field{this->fc.register_real_field(
              "nodal_pt_field", NbComponent, PixelSubDiv::NodalPt)},
          free_pt_field{this->fc.register_real_field(
              "free_pt_field", NbComponent, PixelSubDiv::FreePt,
              Unit::unitless(), NbFreeSubDiv)},
          pixel_map{pixel_field}, quad_pt_map{quad_pt_field},
          nodal_pt_map{nodal_pt_field}, free_pt_map{free_pt_field},
          pixel_quad_pt_map{quad_pt_field, PixelSubDiv::Pixel},
          pixel_nodal_pt_map{nodal_pt_field, PixelSubDiv::Pixel},
          pixel_free_pt_map{free_pt_field, PixelSubDiv::Pixel} {};

    RealField & pixel_field;
    RealField & quad_pt_field;
    RealField & nodal_pt_field;
    RealField & free_pt_field;

    // mapping over their natural subdivision
    FieldMap<Real, Mapping::Mut> pixel_map;
    FieldMap<Real, Mapping::Mut> quad_pt_map;
    FieldMap<Real, Mapping::Mut> nodal_pt_map;
    FieldMap<Real, Mapping::Mut> free_pt_map;

    // mapping over their pixel-aggregate view
    FieldMap<Real, Mapping::Mut> pixel_quad_pt_map;
    FieldMap<Real, Mapping::Mut> pixel_nodal_pt_map;
    FieldMap<Real, Mapping::Mut> pixel_free_pt_map;
  };

}  // namespace muGrid

#endif  // TESTS_LIBMUGRID_FIELD_TEST_FIXTURES_HH_
