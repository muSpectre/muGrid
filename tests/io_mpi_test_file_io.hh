/**
 * @file   io_mpi_test_file_io.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   04 Aug 2020
 *
 * @brief  description
 *
 * Copyright © 2020 Richard Leute
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

#ifndef TESTS_LIBMUGRID_IO_MPI_TEST_FILE_IO_HH_
#define TESTS_LIBMUGRID_IO_MPI_TEST_FILE_IO_HH_

#include "tests.hh"
#include "mpi_context.hh"

#include "libmugrid/field_collection_local.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_map_static.hh"

#include "libmugrid/communicator.hh"

namespace muGrid {
  struct FileIOFixture {
    FileIOFixture()
        : nb_domain_grid_pts{this->spatial_dimension,
                             this->comm.size() *
                                 this->nb_subdomain_grid_pts[1]},
          subdomain_locations{0, this->comm.rank() *
                                     this->nb_subdomain_grid_pts[1]},
          global_fc{this->nb_domain_grid_pts, this->nb_subdomain_grid_pts,
                    this->subdomain_locations, this->nb_sub_pts},
          nb_sub_pts_local{{this->quad, 3}},
          local_fc{this->spatial_dimension, "local_FC", this->nb_sub_pts_local},
          names{"T4_test_field", "T2_test_field", "T1_int_field"},
          t4_field{this->global_fc.register_real_field(
              names[0], muGrid::ipow(this->spatial_dimension, 4), this->quad)},
          t4_field_map{this->t4_field},
          t2_field{this->global_fc.register_real_field(
              names[1], muGrid::ipow(this->spatial_dimension, 2))},
          t2_field_map{this->t2_field},
          t1_field{this->local_fc.register_field<muGrid::Int>(
              names[2], muGrid::ipow(this->spatial_dimension, 1), this->quad)},
          t1_field_map{this->t1_field} {
      // add some pixels to the local field collection
      if (this->comm.size() == 1) {
        for (size_t index = 2; index < 6; index++) {
          this->local_fc.add_pixel(index);
        }
      } else if (this->comm.size() == 2) {
        if (this->comm.rank() == 0) {
          size_t global_offset = 0;
          for (size_t global_index = 2; global_index < 6; global_index++) {
            this->local_fc.add_pixel(global_index - global_offset);
          }
        } else if (this->comm.rank() == 1) {
          size_t global_offset = 6;  // local fc with 6 pixels on rank 0
          for (size_t global_index = 6; global_index < 7; global_index++) {
            this->local_fc.add_pixel(global_index - global_offset);
          }
        }
      }
      this->local_fc.initialise();
    }

    Communicator comm{MPIContext::get_context().comm};
    static constexpr Index_t spatial_dimension{twoD};
    const muGrid::FieldCollection::SubPtMap_t nb_sub_pts{{"quad", 2}};
    const DynCcoord_t nb_subdomain_grid_pts{2, 3};
    const DynCcoord_t nb_domain_grid_pts;
    const DynCcoord_t subdomain_locations;
    const std::string quad{"quad"};
    muGrid::GlobalFieldCollection global_fc;

    const muGrid::FieldCollection::SubPtMap_t nb_sub_pts_local;
    muGrid::LocalFieldCollection local_fc;

    std::vector<std::string> names;  // names for the fields

    // A T4 test field
    muGrid::TypedField<Real> & t4_field;
    muGrid::T4FieldMap<Real, Mapping::Mut, FileIOFixture::spatial_dimension,
                       muGrid::IterUnit::SubPt>
        t4_field_map;

    // A T2 test field
    muGrid::TypedField<Real> & t2_field;
    muGrid::T2FieldMap<Real, Mapping::Mut, FileIOFixture::spatial_dimension,
                       muGrid::IterUnit::SubPt>
        t2_field_map;

    // A local T1 test field
    muGrid::TypedField<int> & t1_field;
    muGrid::T1FieldMap<int, Mapping::Mut, FileIOFixture::spatial_dimension,
                       muGrid::IterUnit::SubPt>
        t1_field_map;
  };
  constexpr Index_t FileIOFixture::spatial_dimension;
}  // namespace muGrid

#endif  // TESTS_LIBMUGRID_IO_MPI_TEST_FILE_IO_HH_
