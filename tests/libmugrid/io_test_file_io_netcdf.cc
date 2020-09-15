/**
 * @file   io_test_file_io_netcdf.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   25 Mai 2020
 *
 * @brief  description
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

#include <stdio.h>

#include <boost/mpl/list.hpp>

#include "io_test_file_io.hh"

#include <libmugrid/file_io_netcdf.hh>
#include <libmugrid/file_io_base.hh>
#include <libmugrid/field_collection_global.hh>
#include <libmugrid/field_typed.hh>
#include <libmugrid/field_map_static.hh>
#include <libmugrid/field_collection.hh>

#include "tests/libmufft/tests.hh"
#include "mpi_context.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(file_io_netcdf);
  BOOST_AUTO_TEST_CASE(NetCDFDimClass) {
    std::string dim_base_name{"test_name"};
    int dim_size{5};
    NetCDFDim netcdf_dim(dim_base_name, dim_size);
  };

  BOOST_AUTO_TEST_CASE(NetCDFVariablesClass) {
    auto & comm{MPIContext::get_context().comm};
    const Index_t spatial_dimension{twoD};
    const muGrid::FieldCollection::SubPtMap_t & nb_sub_pts{{"one", 1}};

    const DynCcoord_t & nb_domain_grid_pts{3, 3};
    const DynCcoord_t & nb_subdomain_grid_pts{3, 3};
    const DynCcoord_t & subdomain_locations{0, 0};
    muGrid::GlobalFieldCollection global_fc(
        spatial_dimension, nb_domain_grid_pts, nb_subdomain_grid_pts,
        subdomain_locations, nb_sub_pts);

    muGrid::LocalFieldCollection local_fc(spatial_dimension, "local_FC",
                                          nb_sub_pts);

    const std::string & unique_name{"T4_test_field"};
    global_fc.register_real_field(unique_name,
                                  muGrid::ipow(spatial_dimension, 4));

    local_fc.register_real_field(unique_name,
                                 muGrid::ipow(spatial_dimension, 3));

    muGrid::Field & global_field{global_fc.get_field(unique_name)};
    muGrid::Field & local_field{local_fc.get_field(unique_name)};

    NetCDFDimensions dimensions;
    std::vector<std::shared_ptr<NetCDFDim>> dim_sp;
    dimensions.add_field_dims_global(global_field, dim_sp);
    dimensions.add_field_dims_local(local_field, dim_sp, comm);

    NetCDFVariables variables;
    variables.add_field_var(global_field, dim_sp);
    variables.add_field_var(local_field, dim_sp);
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFCWrite, FileIOFixture) {
    const std::string file_name{"test_file.nc"};
    auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
    remove(file_name.c_str());  // remove test_file if it already exists

    for (auto && id_val : this->t4_field_map.enumerate_indices()) {
      auto && index{std::get<0>(id_val)};
      auto && value{std::get<1>(id_val)};

      std::vector<Real> fill_value;
      for (int i = 0; i < 16; i++) {
        fill_value.push_back((index * 16 + i) * 2);
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
        fill_value[i] = (index * 4 + i) * 2;
      }
      Eigen::Map<Eigen::Matrix<double, 2, 2>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value);
      value = fill_value_map;
    }

    int fill_value{9};
    for (auto && id_val : this->t1_field_map.enumerate_indices()) {
      auto && value{std::get<1>(id_val)};
      fill_value++;
      int fill_value_vec[2];
      for (int i = 0; i < 2; i++) {
        fill_value_vec[i] = fill_value * (i + 1);
      }
      Eigen::Map<Eigen::Matrix<int, 2, 1>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value_vec);
      value = fill_value_map;
    }

    FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, this->comm);

    file_io_netcdf_w.register_field_collection(this->global_fc);
    file_io_netcdf_w.register_field_collection(this->local_fc);

    // write n_frames frames
    int n_frames{2};
    for (int i = 0; i < n_frames; i++) {
      for (auto && id_val : this->t4_field_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        value *= (i + 1);
      }
      for (auto && id_val : this->t2_field_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        value *= (i + 1);
      }
      file_io_netcdf_w.append_frame().write(this->names);
    }

    // error because you wan to access an frame out of range
    BOOST_CHECK_THROW(file_io_netcdf_w.write(2, names), muGrid::FileIOError);
    // negative frame number
    file_io_netcdf_w.write(-1, this->names);

    // call last frame by [] operator
    file_io_netcdf_w[-1].write(this->names);

    // test no parameter write function of FileFrame
    file_io_netcdf_w[-1].write();

    file_io_netcdf_w.close();
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFRead, FileIOFixture) {
    const std::string file_name{"test_file.nc"};
    auto open_mode_r = muGrid::FileIOBase::OpenMode::Read;

    // test reading of NetCDF file "test_file.nc"
    FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, this->comm);
    file_io_netcdf_r.register_field_collection(this->global_fc);
    file_io_netcdf_r.register_field_collection(this->local_fc);

    // check if the fieds are correct
    // loop over FileIOBase object
    int i_frame{0};  // used to compute reference values
    for (auto & frame : file_io_netcdf_r) {
      frame.read(this->names);
      for (auto && id_val : this->t4_field_map.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 16; i++) {
          Real reference{
              static_cast<Real>((index * 16 + i) * 2 * (i_frame + 1))};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
      i_frame++;
    }

    int n_frames{2};
    for (int frame = 0; frame < n_frames; frame++) {
      file_io_netcdf_r.read(frame, this->names);
      for (auto && id_val : this->t2_field_map.enumerate_indices()) {
        auto && index{std::get<0>(id_val)};
        auto && value{std::get<1>(id_val)};
        for (int i = 0; i < 4; i++) {
          Real reference{static_cast<Real>(index * 4 * 2 + 2 * i) *
                         (frame + 1)};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
      int fill_value{9};
      for (auto && id_val : this->t1_field_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        fill_value++;
        for (int i = 0; i < 2; i++) {
          int reference{fill_value * (i + 1)};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
    }
    file_io_netcdf_r.close();  // close file
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFAppend, FileIOFixture) {
    // fields with other values than in previous write
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

    int fill_value{100};
    for (auto && id_val : this->t1_field_map.enumerate_indices()) {
      auto && value{std::get<1>(id_val)};
      fill_value++;
      int fill_value_vec[2];
      for (int i = 0; i < 2; i++) {
        fill_value_vec[i] = fill_value * (i + 1);
      }
      Eigen::Map<Eigen::Matrix<int, 2, 1>, 0, Eigen::Stride<0, 0>>
          fill_value_map(fill_value_vec);
      value = fill_value_map;
    }

    // test appending of NetCDF file "test_file.nc"
    const std::string file_name{"test_file.nc"};
    auto open_mode_a = muGrid::FileIOBase::OpenMode::Append;
    FileIONetCDF file_io_netcdf_a(file_name, open_mode_a, this->comm);
    file_io_netcdf_a.register_field_collection(this->global_fc);
    file_io_netcdf_a.register_field_collection(this->local_fc);

    // does it know about all current frames?
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
      // read in single frame
      file_io_netcdf_a[frame].read(this->names);

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
          int reference{fill_value * (i + 1)};
          BOOST_CHECK_EQUAL(value(i), reference);
        }
      }
    }
    file_io_netcdf_a.close();  // close file
  };

  BOOST_FIXTURE_TEST_CASE(FileIONetCDFIterator, FileIOFixtureIterator) {
    // write a file with several frames
    const std::string file_name{"test_frames.nc"};
    remove(file_name.c_str());  // remove file if it already exists
    auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
    FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, this->comm);

    file_io_netcdf_w.register_field_collection(this->global_fc);

    for (int i = 0; i < 5; i++) {
      Real fill_value_t1[2];
      fill_value_t1[0] = i;
      fill_value_t1[1] = i;
      Eigen::Map<Eigen::Matrix<Real, 2, 1>, 0, Eigen::Stride<0, 0>>
          fill_value_map_t1(fill_value_t1);
      for (auto && id_val : this->t1_f_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        value = fill_value_map_t1;
      }

      Real fill_value_t2[4];
      fill_value_t2[0] = i;
      fill_value_t2[1] = i;
      fill_value_t2[2] = i;
      fill_value_t2[3] = i;
      Eigen::Map<Eigen::Matrix<Real, 2, 2>, 0, Eigen::Stride<0, 0>>
          fill_value_map_t2(fill_value_t2);
      for (auto && id_val : this->t2_f_map.enumerate_indices()) {
        auto && value{std::get<1>(id_val)};
        value = fill_value_map_t2;
      }

      file_io_netcdf_w.append_frame().write(this->names);
    }

    // iterate over FileIONetCDF object
    // std::cout << "\nbefore loop" << std::endl;
    // const FileFrame & frame{*(file_io_netcdf_w.begin())};
    // frame.read();
    // frame.write();
    for (FileFrame frame : file_io_netcdf_w) {
      frame.read(this->names);
      // TODO(RLeute): check for equality of read frame!
    }
  };

  BOOST_FIXTURE_TEST_CASE(LocalFCSameNames, FileIOFixture) {
    const std::string file_name{"LFC_same_names.nc"};
    remove(file_name.c_str());  // remove file if it already exists
    auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
    FileIONetCDF file_io_netcdf(file_name, open_mode_w, this->comm);

    file_io_netcdf.register_field_collection(this->global_fc);
    file_io_netcdf.register_field_collection(this->local_fc);

    // second Local Field Collection wit same name as first one
    LocalFieldCollection local_fc_2(this->spatial_dimension,
                                    this->local_fc.get_name(),
                                    this->nb_sub_pts_local);
    local_fc_2.register_field<int>(
        "lfc_2_field", muGrid::ipow(this->spatial_dimension, 1), this->quad);
    for (size_t index = 2; index < 7; index++) {
      local_fc_2.add_pixel(index);
    }
    local_fc_2.initialise();

    BOOST_CHECK_THROW(file_io_netcdf.register_field_collection(local_fc_2),
                      muGrid::FieldCollectionError);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
