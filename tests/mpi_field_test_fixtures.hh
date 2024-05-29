/**
 * @file   mpi_field_test_fixtures.hh
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

#ifndef TESTS_LIBMUGRID_MPI_FIELD_TEST_FIXTURES_HH_
#define TESTS_LIBMUGRID_MPI_FIELD_TEST_FIXTURES_HH_

#include "tests.hh"
#include "mpi_context.hh"

#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/field_map_static.hh"

#include "libmugrid/grid_common.hh"
#include "libmugrid/communicator.hh"

namespace muGrid {
  struct MpiFieldMapFixture {
   public:
    Communicator comm{MPIContext::get_context().comm};
    static constexpr Dim_t NbQuadPts{2};
    static constexpr Dim_t NbComponent{2};
    static constexpr Index_t spatial_dimension{threeD};
    const FieldCollection::SubPtMap_t nb_sub_pts{{"quad", NbQuadPts}};
    const DynCcoord_t nb_subdomain_grid_pts{3, 3, 1};
    const DynCcoord_t nb_subdomain_grid_pts_empty{0, 0, 0};
    const std::string quad{"quad"};
  };
  constexpr Dim_t MpiFieldMapFixture::NbQuadPts;
  constexpr Dim_t MpiFieldMapFixture::NbComponent;
  constexpr Index_t MpiFieldMapFixture::spatial_dimension;

  /* ---------------------------------------------------------------------- */
  struct MpiFieldMapFixtureEmptyProcs : public MpiFieldMapFixture {
    MpiFieldMapFixtureEmptyProcs()
        : nb_domain_grid_pts{this->nb_subdomain_grid_pts[0],
                             this->nb_subdomain_grid_pts[1],
                             this->comm.size() > 1
                                 ? (this->comm.size() - 1) *
                                       this->nb_subdomain_grid_pts[2]
                                 : this->nb_subdomain_grid_pts[2]},
          subdomain_locations{
              0, 0, this->comm.rank() * this->nb_subdomain_grid_pts[2]},
          fc{this->nb_domain_grid_pts,
             this->comm.rank() + 1 == this->comm.size() and
                     this->comm.size() > 1
                 ? this->nb_subdomain_grid_pts_empty
                 : this->nb_subdomain_grid_pts,
             this->subdomain_locations, this->nb_sub_pts},
          pixel_field{this->fc.register_real_field("pixel_field", NbComponent,
                                                   PixelTag)},
          quad_pt_field{this->fc.register_real_field("quad_pt_field",
                                                     NbComponent, "quad")},
          pixel_map{pixel_field}, quad_pt_map{quad_pt_field},
          pixel_quad_pt_map{quad_pt_field, IterUnit::Pixel} {};

    const DynCcoord_t nb_domain_grid_pts;
    const DynCcoord_t subdomain_locations;

    GlobalFieldCollection fc;

    RealField & pixel_field;
    RealField & quad_pt_field;

    // mapping over their natural subdivision
    FieldMap<Real, Mapping::Mut> pixel_map;
    FieldMap<Real, Mapping::Mut> quad_pt_map;

    // mapping over their pixel-aggregate view
    FieldMap<Real, Mapping::Mut> pixel_quad_pt_map;
  };

  /* ---------------------------------------------------------------------- */
  struct MpiFieldMapFixtureFullProcs : public MpiFieldMapFixture {
    MpiFieldMapFixtureFullProcs()
        : nb_domain_grid_pts{this->nb_subdomain_grid_pts[0],
                             this->nb_subdomain_grid_pts[1],
                             this->comm.size() *
                                 this->nb_subdomain_grid_pts[2]},
          subdomain_locations{
              0, 0, this->comm.rank() * this->nb_subdomain_grid_pts[2]},
          fc{this->nb_domain_grid_pts, this->nb_subdomain_grid_pts,
             this->subdomain_locations, this->nb_sub_pts},
          pixel_field{this->fc.register_real_field("pixel_field", NbComponent,
                                                   PixelTag)},
          quad_pt_field{this->fc.register_real_field("quad_pt_field",
                                                     NbComponent, "quad")},
          pixel_map{pixel_field}, quad_pt_map{quad_pt_field},
          pixel_quad_pt_map{quad_pt_field, IterUnit::Pixel} {};

    const DynCcoord_t nb_domain_grid_pts;
    const DynCcoord_t subdomain_locations;

    GlobalFieldCollection fc;

    RealField & pixel_field;
    RealField & quad_pt_field;

    // mapping over their natural subdivision
    FieldMap<Real, Mapping::Mut> pixel_map;
    FieldMap<Real, Mapping::Mut> quad_pt_map;

    // mapping over their pixel-aggregate view
    FieldMap<Real, Mapping::Mut> pixel_quad_pt_map;
  };
}  // namespace muGrid

#endif  // TESTS_LIBMUGRID_MPI_FIELD_TEST_FIXTURES_HH_
