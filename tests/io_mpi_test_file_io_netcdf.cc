/**
 * @file   io_mpi_test_file_io_netcdf.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   18 Jun 2020
 *
 * @brief  parallel tests for parallel file IO with PnetCDF
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

#include <stdio.h>

#include <boost/mpl/list.hpp>

#include "mpi_context.hh"

#include "io_mpi_test_file_io.hh"

#include "libmugrid/file_io_netcdf.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_map_static.hh"
#include "libmugrid/grid_common.hh"

using muGrid::operator<<;

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(file_io_netcdf);

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFCWrite, FileIOFixture) {
    const std::string file_name{"test_parallel_file.nc"};
    remove(file_name.c_str());  // remove test_file if it already exists

    // fill the fields with values
    for (auto && id_val : this->t4_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};
      Real fill_value[16];
      for (int i = 0; i < 16; i++) {
        fill_value[i] = (index * muGrid::ipow(this->spatial_dimension, 4) + i) +
                        this->comm.rank() *
                            muGrid::ipow(this->spatial_dimension, 4) *
                            this->nb_subdomain_grid_pts[0] *
                            this->nb_subdomain_grid_pts[1] * 2;
      }
      Eigen::Map<Eigen::Matrix<double, 4, 4>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value);
      value = fill_value_map;
    }
    // wait until all ranks have initialised all values of the field
    MPI_Barrier(this->comm.get_mpi_comm());

    for (auto && id_val : this->t2_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};
      Real fill_value[4];
      for (int i = 0; i < 4; i++) {
        fill_value[i] =
            index * muGrid::ipow(this->spatial_dimension, 2) * 2 + 2 * i +
            this->comm.rank() * muGrid::ipow(this->spatial_dimension, 2) *
                this->nb_subdomain_grid_pts[0] *
                this->nb_subdomain_grid_pts[1] * 2;
      }
      Eigen::Map<Eigen::Matrix<double, 2, 2>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value);
      value = fill_value_map;
    }
    // wait until all ranks have initialised all values of the field
    MPI_Barrier(this->comm.get_mpi_comm());

    int fill_value{9};
    for (auto && id_val : this->t1_field_map.enumerate_indices()) {
      auto && value{std::get<1>(id_val)};
      fill_value++;
      int fill_value_vec[2];
      for (int i = 0; i < 2; i++) {
        fill_value_vec[i] = fill_value * (i + 1) + this->comm.rank() * 100;
      }
      Eigen::Map<Eigen::Matrix<int, 2, 1>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value_vec);
      value = fill_value_map;
    }
    // wait until all ranks have initialised all values of the field
    MPI_Barrier(this->comm.get_mpi_comm());

    auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
    FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, this->comm);
    file_io_netcdf_w.register_field_collection(this->global_fc);
    file_io_netcdf_w.register_field_collection(this->local_fc);
    file_io_netcdf_w.append_frame().write(this->names);  // write frame 0
    file_io_netcdf_w.append_frame().write(this->names);  // write frame 1
    file_io_netcdf_w.close();
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFRead, FileIOFixture) {
    // test reading of NetCDF file "test_file.nc"
    const std::string file_name{"test_parallel_file.nc"};
    auto open_mode_r = muGrid::FileIOBase::OpenMode::Read;
    MPI_Offset frame{0};
    FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, this->comm);
    file_io_netcdf_r.register_field_collection(this->global_fc);
    file_io_netcdf_r.register_field_collection(this->local_fc);
    file_io_netcdf_r.read(frame, this->names);

    // check if the fields are correct
    for (auto && id_val : this->t4_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};
      for (int i = 0; i < 16; i++) {
        Real reference{static_cast<Real>(
            index * muGrid::ipow(this->spatial_dimension, 4) + i +
            this->comm.rank() * muGrid::ipow(this->spatial_dimension, 4) *
                this->nb_subdomain_grid_pts[0] *
                this->nb_subdomain_grid_pts[1] * 2)};
        BOOST_CHECK_EQUAL(value(i), reference);
      }
    }

    for (auto && id_val : this->t2_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};
      for (int i = 0; i < 4; i++) {
        Real reference{static_cast<Real>(
            index * muGrid::ipow(this->spatial_dimension, 2) * 2 + 2 * i +
            this->comm.rank() * muGrid::ipow(this->spatial_dimension, 2) *
                this->nb_subdomain_grid_pts[0] *
                this->nb_subdomain_grid_pts[1] * 2)};
        BOOST_CHECK_EQUAL(value(i), reference);
      }
    }

    int fill_value{9};
    for (auto && id_val : this->t1_field_map.enumerate_indices()) {
      auto && value{std::get<1>(id_val)};
      fill_value++;
      for (int i = 0; i < 2; i++) {
        int reference = fill_value * (i + 1) + this->comm.rank() * 100;
        BOOST_CHECK_EQUAL(value(i), reference);
      }
    }
    file_io_netcdf_r.close();  // close file
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFAppend, FileIOFixture) {
    // fill the fields with other values than in previous write
    for (auto && id_val : this->t4_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};

      std::vector<Real> fill_value;
      for (int i = 0; i < 16; i++) {
        fill_value.push_back(static_cast<Real>(index * 16 + i) / 4);
      }

      Eigen::Map<Eigen::Matrix<double, 4, 4>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value.data());
      value = fill_value_map;
    }

    for (auto && id_val : this->t2_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};
      Real fill_value[4];
      for (int i = 0; i < 4; i++) {
        fill_value[i] = static_cast<Real>(index * 4 * 2 + 2 * i) / 2;
      }
      Eigen::Map<Eigen::Matrix<double, 2, 2>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value);
      value = fill_value_map;
    }

    // A local T1 test field
    int fill_value{100};
    for (auto && id_val : this->t1_field_map.enumerate_indices()) {
      auto && value{std::get<1>(id_val)};
      fill_value++;
      int fill_value_vec[2];
      for (int i = 0; i < 2; i++) {
        fill_value_vec[i] = fill_value * (i + 1) + this->comm.rank() * 100;
      }
      Eigen::Map<Eigen::Matrix<int, 2, 1>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value_vec);
      value = fill_value_map;
    }

    // test appending of NetCDF file "test_parallel_file.nc"
    const std::string file_name{"test_parallel_file.nc"};
    auto open_mode_a = muGrid::FileIOBase::OpenMode::Append;
    FileIONetCDF file_io_netcdf_a(file_name, open_mode_a, this->comm);
    file_io_netcdf_a.register_field_collection(this->global_fc);
    file_io_netcdf_a.register_field_collection(this->local_fc);

    // does it know about all current frames
    Index_t n_frames{2};
    BOOST_CHECK_EQUAL(file_io_netcdf_a.size(), n_frames);

    // test overwrite and append!
    Index_t frame{0};
    file_io_netcdf_a.append_frame().write(this->names);  // should append
    file_io_netcdf_a.write(frame, this->names);          // should overwrite

    // check if the fields are correct
    std::vector<Index_t> check_frames{
        frame, n_frames};  // overwritten and appended frame
    for (int frame : check_frames) {
      file_io_netcdf_a.read(frame, this->names);
      for (auto && id_val : this->t4_field_map.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 16; i++) {
          Real reference{static_cast<Real>(index * 16 + i) / 4};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }

      for (auto && id_val : this->t2_field_map.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 4; i++) {
          Real reference{static_cast<Real>(index * 4 * 2 + 2 * i) / 2};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }

      int fill_value{100};
      for (auto && id_val : this->t1_field_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        fill_value++;
        for (int i = 0; i < 2; i++) {
          int reference = fill_value * (i + 1) + this->comm.rank() * 100;
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
    }
    file_io_netcdf_a.close();  // close file
  };

  // TODO(RLeute): read with a different number of processors than the file was
  // written, especially for local fields (e.g. write 2 procs read 1 and 3
  // procs)!!!
  BOOST_AUTO_TEST_CASE(VaryingNumberOfProcs) {
    // read/write with different number of processors (1 proc and 2 proc)
    // in total the following operations are done on the same file (1p means
    // operation done with one processor 2p with two processors):
    // write(2p) -> read(1p) -> append(1p) -> write(1p) -> read(2p) ->
    // append(2p) -> read(1p)

    auto & comm_2{MPIContext::get_context().comm};
    if (comm_2.rank() == 0) {
      std::cout << "\nVaryingNumberOfProcs" << std::endl;
    }

    if (comm_2.size() >= 2) {
      MPI_Comm newcomm;
      MPI_Comm_split(comm_2.get_mpi_comm(), comm_2.rank(), comm_2.rank(),
                     &newcomm);
      Communicator comm_1(newcomm);

      const std::string file_name{"test_varying_num_procs.nc"};
      remove(file_name.c_str());  // remove test file if it already exists
      auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
      auto open_mode_r = muGrid::FileIOBase::OpenMode::Read;
      auto open_mode_a = muGrid::FileIOBase::OpenMode::Append;

      // global field collection
      const Index_t & spatial_dimension{2};
      const Dim_t Dim{twoD};
      const std::string quad{"quad"};
      const muGrid::FieldCollection::SubPtMap_t & nb_sub_pts{{quad, 2}};
      const DynCcoord_t & nb_subdomain_grid_pts_2{2, 3};
      const DynCcoord_t & nb_subdomain_grid_pts_1{2, 6};
      const DynCcoord_t & nb_domain_grid_pts{2, 6};
      const DynCcoord_t & subdomain_locations_2{
          0, comm_2.rank() * nb_subdomain_grid_pts_2[1]};
      const DynCcoord_t & subdomain_locations_1{0, 0};
      muGrid::GlobalFieldCollection global_fc_2(
          nb_domain_grid_pts, nb_subdomain_grid_pts_2, subdomain_locations_2,
          nb_sub_pts);
      muGrid::GlobalFieldCollection global_fc_1(
          nb_domain_grid_pts, nb_subdomain_grid_pts_1, subdomain_locations_1,
          nb_sub_pts);
      muGrid::TypedField<Real> & t2_field_2{global_fc_2.register_real_field(
          "T2_test_field", muGrid::ipow(spatial_dimension, 2))};
      muGrid::T2FieldMap<Real, Mapping::Mut, Dim, muGrid::IterUnit::SubPt>
          t2_field_map_2{t2_field_2};
      muGrid::TypedField<Real> & t2_field_1{global_fc_1.register_real_field(
          "T2_test_field", muGrid::ipow(spatial_dimension, 2))};
      muGrid::T2FieldMap<Real, Mapping::Mut, Dim, muGrid::IterUnit::SubPt>
          t2_field_map_1{t2_field_1};

      // local field collection
      const muGrid::FieldCollection::SubPtMap_t & nb_sub_pts_local{{"quad", 3}};
      muGrid::LocalFieldCollection local_fc_2(spatial_dimension, "local_FC",
                                              nb_sub_pts_local);
      muGrid::LocalFieldCollection local_fc_1(spatial_dimension, "local_FC",
                                              nb_sub_pts_local);
      if (comm_2.rank() == 0) {
        size_t global_offset = 0;
        for (size_t global_index = 2; global_index < 6; global_index++) {
          local_fc_2.add_pixel(global_index - global_offset);
        }
      } else if (comm_2.rank() == 1) {
        size_t global_offset = 6;  // local fc with 6 pixels on rank 0
        for (size_t global_index = 6; global_index < 8; global_index++) {
          local_fc_2.add_pixel(global_index - global_offset);
        }
      }
      local_fc_2.initialise();

      if (comm_1.rank() == 0) {
        for (size_t index = 2; index < 8; index++) {
          local_fc_1.add_pixel(index);
        }
        local_fc_1.initialise();
      }

      muGrid::TypedField<int> & t1_field_2{
          local_fc_2.register_field<muGrid::Int>(
              "T1_int_field", muGrid::ipow(spatial_dimension, 1), quad)};
      muGrid::T1FieldMap<int, Mapping::Mut, Dim, muGrid::IterUnit::SubPt>
          t1_field_map_2{t1_field_2};

      muGrid::TypedField<int> & t1_field_1{
          local_fc_1.register_field<muGrid::Int>(
              "T1_int_field", muGrid::ipow(spatial_dimension, 1), quad)};
      muGrid::T1FieldMap<int, Mapping::Mut, Dim, muGrid::IterUnit::SubPt>
          t1_field_map_1{t1_field_1};

      // write with comm_2 (two procs) ------------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "write(2p)" << std::endl;
      }
      // fill values
      for (auto && id_val : t2_field_map_2.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        Real fill_value[4];
        for (int i = 0; i < 4; i++) {
          fill_value[i] =
              index * muGrid::ipow(spatial_dimension, 2) * 2 + 2 * i +
              comm_2.rank() * muGrid::ipow(spatial_dimension, 2) *
                  nb_subdomain_grid_pts_2[0] * nb_subdomain_grid_pts_2[1] * 2;
        }
        Eigen::Map<Eigen::Matrix<double, 2, 2>, 0, Eigen::Stride<0, 0>>
            fill_value_map(fill_value);
        value = fill_value_map;
      }
      // wait until all ranks have initialised all values of the field
      MPI_Barrier(comm_2.get_mpi_comm());
      int fill_value_2{9};
      for (auto && id_val : t1_field_map_2.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        fill_value_2++;
        int fill_value_vec[2];
        for (int i = 0; i < 2; i++) {
          fill_value_vec[i] = fill_value_2 * (i + 1) + comm_2.rank() * 100;
        }
        Eigen::Map<Eigen::Matrix<int, 2, 1>, 0, Eigen::Stride<0, 0>>
            fill_value_map(fill_value_vec);
        value = fill_value_map;
      }
      // wait until all ranks have initialised all values of the field
      MPI_Barrier(comm_2.get_mpi_comm());

      FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, comm_2);
      file_io_netcdf_w.register_field_collection(global_fc_2);
      file_io_netcdf_w.register_field_collection(local_fc_2);
      file_io_netcdf_w.append_frame().write();  // write frame 0
      file_io_netcdf_w.close();

      // read with comm_1 (one processor) ---------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "read(1p)" << std::endl;
        FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, comm_1);
        file_io_netcdf_r.register_field_collection(global_fc_1);
        file_io_netcdf_r.register_field_collection(local_fc_1);
        Index_t n_frames_r{1};
        BOOST_CHECK_EQUAL(file_io_netcdf_r.size(), n_frames_r);
        file_io_netcdf_r.read(0);  // read frame 0
        // check values
        for (auto && id_val : t2_field_map_1.enumerate_indices()) {
          auto && index{std::get<0>(id_val)};
          auto && value{std::get<1>(id_val)};
          for (int i = 0; i < 4; i++) {
            Real reference{static_cast<Real>(
                index * muGrid::ipow(spatial_dimension, 2) * 2 + 2 * i +
                comm_1.rank() * muGrid::ipow(spatial_dimension, 2) *
                    nb_subdomain_grid_pts_1[0] * nb_subdomain_grid_pts_1[1] *
                    2)};
            BOOST_CHECK_EQUAL(value(i), reference);
          }
        }
        int fill_value_1{9};
        for (auto && id_val : t1_field_map_1.enumerate_indices()) {
          auto && index{std::get<0>(id_val)};
          auto && value{std::get<1>(id_val)};
          if (index == 18) {
            fill_value_1 = 9;  // memic the initialisation on two procs
                               // (initialise fill_value_1 a second time)
          }
          fill_value_1++;
          for (int i = 0; i < 2; i++) {
            int reference = fill_value_1 * (i + 1) + (index >= 18 ? 100 : 0);
            BOOST_CHECK_EQUAL(value(i), reference);
          }
        }
        file_io_netcdf_r.close();
      }

      // append with comm_1 (one processor) -------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "append(1p)" << std::endl;
        FileIONetCDF file_io_netcdf_a(file_name, open_mode_a, comm_1);
        // does it know about all current frames
        Index_t n_frames_a{1};
        BOOST_CHECK_EQUAL(file_io_netcdf_a.size(), n_frames_a);
        file_io_netcdf_a.register_field_collection(global_fc_1);
        file_io_netcdf_a.register_field_collection(local_fc_1);
        file_io_netcdf_a.append_frame().write();  // write frame 1
      }

      // write with comm_1 (one processor) --------------------------------- //
      if (comm_2.rank() == 0) {
        // you can only write a file once afterwards you can only append or read
        std::cout << "write(1p) - not possible to write two times the same file"
                  << std::endl;
        BOOST_CHECK_THROW(
            FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, comm_1),
            std::runtime_error);
      }

      MPI_Barrier(comm_2.get_mpi_comm());
      MPI_Barrier(comm_1.get_mpi_comm());

      // read with comm_2 (two processors) --------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "read(2p)" << std::endl;
      }
      FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, comm_2);
      file_io_netcdf_r.register_field_collection(global_fc_2);
      file_io_netcdf_r.register_field_collection(local_fc_2);
      Index_t n_frames_r{2};
      BOOST_CHECK_EQUAL(file_io_netcdf_r.size(), n_frames_r);
      file_io_netcdf_r.read(1);  // read frame 1
      // check values
      for (auto && id_val : t2_field_map_2.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 4; i++) {
          Real reference{static_cast<Real>(
              index * muGrid::ipow(spatial_dimension, 2) * 2 + 2 * i +
              comm_2.rank() * muGrid::ipow(spatial_dimension, 2) *
                  nb_subdomain_grid_pts_2[0] * nb_subdomain_grid_pts_2[1] * 2)};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
      int fill_value_1{9};
      for (auto && id_val : t1_field_map_2.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        if (index == 18) {
          fill_value_1 = 9;  // memic the initialisation on two procs
                             // (initialise fill_value_1 a second time)
        }
        fill_value_1++;
        for (int i = 0; i < 2; i++) {
          int reference = fill_value_1 * (i + 1) + comm_2.rank() * 100;
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
      file_io_netcdf_r.close();

      // append with comm_2 (two processors) ------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "append(2p)" << std::endl;
      }

      // change field values
      for (auto && id_val : t2_field_map_2.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 4; i++) {
          value(i) = 20 + comm_2.rank() + 1;
        }
      }
      for (auto && id_val : t1_field_map_2.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 2; i++) {
          value(i) = 20 + comm_2.rank() + 1;
        }
      }

      FileIONetCDF file_io_netcdf_a(file_name, open_mode_a, comm_2);
      // does it know about all current frames
      Index_t n_frames_a{2};
      BOOST_CHECK_EQUAL(file_io_netcdf_a.size(), n_frames_a);
      file_io_netcdf_a.register_field_collection(global_fc_2);
      file_io_netcdf_a.register_field_collection(local_fc_2);
      file_io_netcdf_a.append_frame().write();  // write frame 2

      MPI_Barrier(comm_2.get_mpi_comm());
      MPI_Barrier(comm_1.get_mpi_comm());

      // read with comm_1 (one processor) ---------------------------------- //
      if (comm_2.rank() == 0) {
        std::cout << "read(1p)" << std::endl;
        FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, comm_1);
        file_io_netcdf_r.register_field_collection(global_fc_1);
        file_io_netcdf_r.register_field_collection(local_fc_1);
        Index_t n_frames_r{3};
        BOOST_CHECK_EQUAL(file_io_netcdf_r.size(), n_frames_r);
        file_io_netcdf_r.read(2);  // read frame 2
        // check values
        for (auto && id_val : t2_field_map_1.enumerate_indices()) {
          auto && index{std::get<0>(id_val)};
          auto && value{std::get<1>(id_val)};
          Real reference = 20 + (index >= 6 ? 1 : 0) + 1;
          for (int i = 0; i < 4; i++) {
            BOOST_CHECK_EQUAL(value(i), reference);
          }
        }
        for (auto && id_val : t1_field_map_1.enumerate_indices()) {
          auto && index{std::get<0>(id_val)};
          auto && value{std::get<1>(id_val)};
          int reference = 20 + (index >= 18 ? 1 : 0) + 1;
          for (int i = 0; i < 2; i++) {
            BOOST_CHECK_EQUAL(value(i), reference);
          }
        }
        file_io_netcdf_r.close();
      }
    }  // end comm_2.size() >= 2
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
