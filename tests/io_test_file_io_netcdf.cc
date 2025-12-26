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

#include "io/file_io_netcdf.hh"
#include "io/file_io_base.hh"
#include "collection/field_collection_global.hh"
#include "field/field_typed.hh"
#include "field/field_map_static.hh"
#include "collection/field_collection.hh"
#include "field/state_field.hh"

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

    const DynGridIndex & nb_domain_grid_pts{3, 3};
    const DynGridIndex & nb_subdomain_grid_pts{3, 3};
    const DynGridIndex & subdomain_locations{0, 0};
    muGrid::GlobalFieldCollection global_fc(nb_domain_grid_pts,
                                            nb_subdomain_grid_pts,
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

  BOOST_FIXTURE_TEST_CASE(WriteStateFields, FileIOFixture) {
    const std::string file_name{"test_statefields.nc"};
    remove(file_name.c_str());  // remove file if it already exists
    auto open_mode = muGrid::FileIOBase::OpenMode::Write;
    FileIONetCDF file_io_netcdf(file_name, open_mode, this->comm);

    // construct a statefield in the global and local field collection
    const std::string glob_sf_name{"glob_state_field"};
    const std::string loc_sf_name{"local_state_field"};
    const Index_t glob_nb_mem{3};
    const Index_t loc_nb_mem{1};
    Index_t nb_components{};
    std::string sub_division{};
    for (auto & element : this->nb_sub_pts) {
      nb_components = element.second;
      sub_division = element.first;
    }
    TypedStateField<Real> & glob_state_field{
        this->global_fc.register_state_field<Real>(
            glob_sf_name, glob_nb_mem, nb_components, sub_division)};

    TypedStateField<int> & loc_state_field{
        this->local_fc.register_state_field<int>(loc_sf_name, loc_nb_mem,
                                                 nb_components, sub_division)};

    for (int i = 1; i <= glob_nb_mem + 1; i++) {
      // fill values into the global state field
      TypedField<Real> & current_state_field{glob_state_field.current()};
      current_state_field.eigen_vec().setConstant(glob_nb_mem + 1 - i);
      glob_state_field.cycle();
    }
    // reorder state field such that its values are increasing in history
    // (0,1,2,3)
    glob_state_field.cycle();
    glob_state_field.cycle();
    glob_state_field.cycle();

    for (int i = 1; i <= loc_nb_mem + 1; i++) {
      // fill values into the local state field
      TypedField<int> & current_state_field{loc_state_field.current()};
      current_state_field.eigen_vec().setConstant(loc_nb_mem + 1 - i);
      loc_state_field.cycle();
    }

    std::vector<std::string> field_names{};
    std::vector<std::string> state_field_unique_prefixes{glob_sf_name};
    file_io_netcdf.register_field_collection(this->global_fc, field_names,
                                             state_field_unique_prefixes);
    file_io_netcdf.register_field_collection(this->local_fc, field_names);

    // double registration raises error
    BOOST_CHECK_THROW(file_io_netcdf.register_field_collection(this->global_fc),
                      muGrid::FileIOError);

    // write frame 0
    file_io_netcdf.append_frame().write();

    // change values of global and local state field + cycle
    for (int i = 1; i <= glob_nb_mem + 1; i++) {
      // fill values into the global state field
      TypedField<Real> & current_state_field{glob_state_field.current()};
      current_state_field.eigen_vec().setConstant(glob_nb_mem + 5 - i);
      glob_state_field.cycle();
    }
    glob_state_field.cycle();
    glob_state_field.cycle();
    glob_state_field.cycle();

    for (int i = 1; i <= loc_nb_mem + 1; i++) {
      // fill values into the local state field
      TypedField<int> & current_state_field{loc_state_field.current()};
      current_state_field.eigen_vec().setConstant(loc_nb_mem + 3 - i);
      loc_state_field.cycle();
    }

    loc_state_field.cycle();

    // write frame 2
    file_io_netcdf.append_frame().write();
    file_io_netcdf.close();
  }

  BOOST_FIXTURE_TEST_CASE(AppendStateFields, FileIOFixture) {
    const std::string file_name{"test_statefields.nc"};
    auto open_mode = muGrid::FileIOBase::OpenMode::Append;
    FileIONetCDF file_io_netcdf(file_name, open_mode, this->comm);

    // construct a statefield in the global and local field collection
    const std::string glob_sf_name{"glob_state_field"};
    const std::string loc_sf_name{"local_state_field"};
    const Index_t glob_nb_mem{3};
    const Index_t loc_nb_mem{1};
    Index_t nb_components{};
    std::string sub_division{};
    for (auto & element : this->nb_sub_pts) {
      nb_components = element.second;
      sub_division = element.first;
    }
    TypedStateField<Real> & glob_state_field{
        this->global_fc.register_state_field<Real>(
            glob_sf_name, glob_nb_mem, nb_components, sub_division)};

    TypedStateField<int> & loc_state_field{
        this->local_fc.register_state_field<int>(loc_sf_name, loc_nb_mem,
                                                 nb_components, sub_division)};

    std::vector<std::string> field_names{};
    std::vector<std::string> state_field_unique_prefixes{glob_sf_name};
    file_io_netcdf.register_field_collection(this->global_fc, field_names,
                                             state_field_unique_prefixes);
    file_io_netcdf.register_field_collection(this->local_fc, field_names);

    for (int i = 1; i <= glob_nb_mem + 1; i++) {
      // fill values into the global state field
      TypedField<Real> & current_state_field{glob_state_field.current()};
      current_state_field.eigen_vec().setConstant(glob_nb_mem + 9 - i);
      glob_state_field.cycle();
    }
    glob_state_field.cycle();

    for (int i = 1; i <= loc_nb_mem + 1; i++) {
      // fill values into the local state field
      TypedField<int> & current_state_field{loc_state_field.current()};
      current_state_field.eigen_vec().setConstant(loc_nb_mem + 5 - i);
      loc_state_field.cycle();
    }
    loc_state_field.cycle();

    file_io_netcdf.append_frame().write();  // append frame 2
    file_io_netcdf.close();
  }

  BOOST_FIXTURE_TEST_CASE(ReadStateFields, FileIOFixture) {
    const std::string file_name{"test_statefields.nc"};
    auto open_mode = muGrid::FileIOBase::OpenMode::Read;
    FileIONetCDF file_io_netcdf(file_name, open_mode, this->comm);

    // construct a statefield in the global and local field collection
    const std::string glob_sf_name{"glob_state_field"};
    const std::string loc_sf_name{"local_state_field"};
    const Index_t glob_nb_mem{3};
    const Index_t loc_nb_mem{1};
    Index_t nb_components{};
    std::string sub_division{};
    for (auto & element : this->nb_sub_pts) {
      nb_components = element.second;
      sub_division = element.first;
    }
    TypedStateField<Real> & glob_state_field{
        this->global_fc.register_state_field<Real>(
            glob_sf_name, glob_nb_mem, nb_components, sub_division)};

    TypedStateField<int> & loc_state_field{
        this->local_fc.register_state_field<int>(loc_sf_name, loc_nb_mem,
                                                 nb_components, sub_division)};

    std::vector<std::string> field_names{};
    std::vector<std::string> state_field_unique_prefixes{glob_sf_name};
    file_io_netcdf.register_field_collection(this->global_fc, field_names,
                                             state_field_unique_prefixes);
    file_io_netcdf.register_field_collection(this->local_fc, field_names);

    // cycle the global state field to see if it is coorect read in
    glob_state_field.cycle();
    glob_state_field.cycle();

    // check reading frame 0
    std::vector<std::string> read_field_names{glob_sf_name, loc_sf_name};
    file_io_netcdf.read(0, read_field_names);

    BOOST_CHECK_EQUAL(glob_state_field.current().eigen_vec()(0, 0), 0);
    BOOST_CHECK_EQUAL(glob_state_field.old(1).eigen_vec()(0, 0), 1);
    BOOST_CHECK_EQUAL(glob_state_field.old(2).eigen_vec()(0, 0), 2);
    BOOST_CHECK_EQUAL(glob_state_field.old(3).eigen_vec()(0, 0), 3);

    BOOST_CHECK_EQUAL(loc_state_field.current().eigen_vec()(0, 0), 1);
    BOOST_CHECK_EQUAL(loc_state_field.old(1).eigen_vec()(0, 0), 0);

    // check reading frame 1
    file_io_netcdf.read(1);
    BOOST_CHECK_EQUAL(glob_state_field.current().eigen_vec()(0, 0), 4);
    BOOST_CHECK_EQUAL(glob_state_field.old(1).eigen_vec()(0, 0), 5);
    BOOST_CHECK_EQUAL(glob_state_field.old(2).eigen_vec()(0, 0), 6);
    BOOST_CHECK_EQUAL(glob_state_field.old(3).eigen_vec()(0, 0), 7);

    BOOST_CHECK_EQUAL(loc_state_field.current().eigen_vec()(0, 0), 2);
    BOOST_CHECK_EQUAL(loc_state_field.old(1).eigen_vec()(0, 0), 3);

    // check reading frame 2
    file_io_netcdf.read(2);
    BOOST_CHECK_EQUAL(glob_state_field.current().eigen_vec()(0, 0), 10);
    BOOST_CHECK_EQUAL(glob_state_field.old(1).eigen_vec()(0, 0), 11);
    BOOST_CHECK_EQUAL(glob_state_field.old(2).eigen_vec()(0, 0), 8);
    BOOST_CHECK_EQUAL(glob_state_field.old(3).eigen_vec()(0, 0), 9);

    BOOST_CHECK_EQUAL(loc_state_field.current().eigen_vec()(0, 0), 4);
    BOOST_CHECK_EQUAL(loc_state_field.old(1).eigen_vec()(0, 0), 5);

    file_io_netcdf.close();
  }

  BOOST_FIXTURE_TEST_CASE(GlobalAttributesWriteMode, FileIOFixture) {
    // WRITE-MODE
    const std::string file_name{"test_global_attributes.nc"};
    remove(file_name.c_str());  // remove file if it already exists
    auto open_mode_w = muGrid::FileIOBase::OpenMode::Write;
    FileIONetCDF file_io_netcdf_w(file_name, open_mode_w, this->comm);
    file_io_netcdf_w.write_global_attribute(this->global_att_1_name,
                                            this->global_att_1_value);

    // register something else than global attributes, a NetCDF file with only
    // global attributes is empty.
    file_io_netcdf_w.register_field_collection(this->global_fc);

    // error for double registration of global attributes
    BOOST_CHECK_THROW(file_io_netcdf_w.write_global_attribute(
                          this->global_att_1_name, this->global_att_1_value),
                      muGrid::FileIOError);

    // check registration of global att after field registration
    file_io_netcdf_w.write_global_attribute(this->global_att_2_name,
                                            this->global_att_2_value);

    // error registration after write()
    file_io_netcdf_w.append_frame().write();
    BOOST_CHECK_THROW(file_io_netcdf_w.write_global_attribute(
                          this->global_att_2_name, this->global_att_2_value),
                      muGrid::FileIOError);

    // check read_global_att_names()
    std::vector<std::string> global_att_names_ref{
        this->global_att_names_default};
    global_att_names_ref.push_back(global_att_1_name);
    global_att_names_ref.push_back(global_att_2_name);
    const std::vector<std::string> global_att_names{
        file_io_netcdf_w.read_global_attribute_names()};
    BOOST_CHECK_EQUAL(global_att_names.size(), global_att_names_ref.size());
    for (size_t i{0}; i < global_att_names.size(); i++) {
      BOOST_CHECK_EQUAL(global_att_names[i], global_att_names_ref[i]);
    }
    // check read_global_att()
    BOOST_CHECK(file_io_netcdf_w.read_global_attribute(this->global_att_1_name)
                    .equal_value(this->global_att_1_value.data()));
    BOOST_CHECK(file_io_netcdf_w.read_global_attribute(this->global_att_2_name)
                    .equal_value(this->global_att_2_value.data()));

    file_io_netcdf_w.close();
  }

  BOOST_FIXTURE_TEST_CASE(GlobalAttributesReadMode, FileIOFixture) {
    // READ-MODE
    const std::string file_name{"test_global_attributes.nc"};
    auto open_mode_r = muGrid::FileIOBase::OpenMode::Read;
    FileIONetCDF file_io_netcdf_r(file_name, open_mode_r, this->comm);
    std::vector<std::string> global_att_names_ref{
        this->global_att_names_default};
    global_att_names_ref.push_back(global_att_1_name);
    global_att_names_ref.push_back(global_att_2_name);
    const std::vector<std::string> global_att_names{
        file_io_netcdf_r.read_global_attribute_names()};
    BOOST_CHECK_EQUAL(global_att_names.size(), global_att_names_ref.size());
    for (size_t i{0}; i < global_att_names.size(); i++) {
      BOOST_CHECK_EQUAL(global_att_names[i], global_att_names_ref[i]);
    }
    for (auto & g_att_name : global_att_names) {
      file_io_netcdf_r.read_global_attribute(g_att_name);
    }
  }

  BOOST_FIXTURE_TEST_CASE(GlobalAttributesAppendMode, FileIOFixture) {
    // APPEND-MODE
    const std::string file_name{"test_global_attributes.nc"};
    auto open_mode_a = muGrid::FileIOBase::OpenMode::Append;
    FileIONetCDF file_io_netcdf_r(file_name, open_mode_a, this->comm);
    std::vector<std::string> global_att_names_ref{
        this->global_att_names_default};
    global_att_names_ref.push_back(global_att_1_name);
    global_att_names_ref.push_back(global_att_2_name);
    const std::vector<std::string> global_att_names{
        file_io_netcdf_r.read_global_attribute_names()};
    BOOST_CHECK_EQUAL(global_att_names.size(), global_att_names_ref.size());
    for (size_t i{0}; i < global_att_names.size(); i++) {
      BOOST_CHECK_EQUAL(global_att_names[i], global_att_names_ref[i]);
    }
    for (auto & g_att_name : global_att_names) {
      file_io_netcdf_r.read_global_attribute(g_att_name);
    }
  }

  // Tests for Phase 1 refactoring improvements
  BOOST_AUTO_TEST_CASE(NetCDFAttVariantStorage) {
    // Test 1.2: Variant-based attribute storage
    // Verify that all attribute types are correctly stored and retrieved

    // Test char vector
    std::vector<char> char_values{'h', 'e', 'l', 'l', 'o'};
    muGrid::NetCDFAtt att_char("test_char", char_values);
    BOOST_CHECK_EQUAL(att_char.get_data_type(), muGrid::MU_NC_CHAR);
    BOOST_CHECK_EQUAL(att_char.get_nelems(), 5);
    // Verify individual elements instead of comparing vectors
    const auto & retrieved_char = att_char.get_typed_value_c();
    BOOST_CHECK_EQUAL(retrieved_char[0], 'h');
    BOOST_CHECK_EQUAL(retrieved_char.size(), size_t(5));

    // Test int vector
    std::vector<muGrid::Int> int_values{1, 2, 3, 4, 5};
    muGrid::NetCDFAtt att_int("test_int", int_values);
    BOOST_CHECK_EQUAL(att_int.get_data_type(), muGrid::MU_NC_INT);
    BOOST_CHECK_EQUAL(att_int.get_nelems(), 5);
    const auto & retrieved_int = att_int.get_typed_value_i();
    BOOST_CHECK_EQUAL(retrieved_int[0], 1);
    BOOST_CHECK_EQUAL(retrieved_int.size(), size_t(5));

    // Test uint vector
    std::vector<muGrid::Uint> uint_values{10, 20, 30};
    muGrid::NetCDFAtt att_uint("test_uint", uint_values);
    BOOST_CHECK_EQUAL(att_uint.get_data_type(), muGrid::MU_NC_UINT);
    BOOST_CHECK_EQUAL(att_uint.get_nelems(), 3);
    const auto & retrieved_uint = att_uint.get_typed_value_ui();
    BOOST_CHECK_EQUAL(retrieved_uint[0], 10u);
    BOOST_CHECK_EQUAL(retrieved_uint.size(), size_t(3));

    // Test Index_t vector
    std::vector<muGrid::Index_t> index_values{100, 200, 300, 400};
    muGrid::NetCDFAtt att_index("test_index", index_values);
    BOOST_CHECK_EQUAL(att_index.get_data_type(), muGrid::MU_NC_INDEX_T);
    BOOST_CHECK_EQUAL(att_index.get_nelems(), 4);
    const auto & retrieved_index = att_index.get_typed_value_l();
    BOOST_CHECK_EQUAL(retrieved_index[0], 100);
    BOOST_CHECK_EQUAL(retrieved_index.size(), size_t(4));

    // Test Real vector
    std::vector<muGrid::Real> real_values{1.5, 2.5, 3.5};
    muGrid::NetCDFAtt att_real("test_real", real_values);
    BOOST_CHECK_EQUAL(att_real.get_data_type(), muGrid::MU_NC_REAL);
    BOOST_CHECK_EQUAL(att_real.get_nelems(), 3);
    const auto & retrieved_real = att_real.get_typed_value_d();
    BOOST_CHECK_EQUAL(retrieved_real[0], 1.5);
    BOOST_CHECK_EQUAL(retrieved_real.size(), size_t(3));
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttVariantDataSize) {
    // Test that variant-based storage correctly computes data sizes
    std::vector<char> char_vals{'a', 'b', 'c'};
    muGrid::NetCDFAtt att_char("test", char_vals);
    BOOST_CHECK_EQUAL(att_char.get_data_size(), 3 * sizeof(char));

    std::vector<muGrid::Real> real_vals{1.0, 2.0};
    muGrid::NetCDFAtt att_real("test", real_vals);
    BOOST_CHECK_EQUAL(att_real.get_data_size(), 2 * sizeof(muGrid::Real));
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttVariantGetValue) {
    // Test that get_value() correctly returns void pointer to variant data
    std::vector<muGrid::Int> int_values{42, 84};
    muGrid::NetCDFAtt att("test", int_values);

    const void * ptr = att.get_value();
    const muGrid::Int * int_ptr = static_cast<const muGrid::Int *>(ptr);
    BOOST_CHECK_EQUAL(int_ptr[0], 42);
    BOOST_CHECK_EQUAL(int_ptr[1], 84);
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttVariantGetValueAsString) {
    // Test that get_value_as_string() works with variant storage
    // Use a string directly which has a built-in vector serialization
    std::string test_str{"test_value"};
    muGrid::NetCDFAtt att("test", test_str);
    std::string str_value = att.get_value_as_string();
    // String formatting produces output, just verify non-empty
    BOOST_CHECK(!str_value.empty());
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttVariantStringConstructor) {
    // Test convenience constructor for strings using variant
    std::string test_string{"hello world"};
    muGrid::NetCDFAtt att("test", test_string);
    BOOST_CHECK_EQUAL(att.get_data_type(), muGrid::MU_NC_CHAR);
    BOOST_CHECK_EQUAL(att.get_nelems(), test_string.size());

    // String should be stored as char vector
    const std::vector<char> & char_vals = att.get_typed_value_c();
    std::string retrieved(char_vals.begin(), char_vals.end());
    BOOST_CHECK_EQUAL(retrieved, test_string);
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttVariantEqualValue) {
    // Test that equal_value() works correctly with variant storage
    std::vector<muGrid::Real> real_values{1.5, 2.5, 3.5};
    muGrid::NetCDFAtt att("test", real_values);

    // Create matching void pointer - must have same size and values
    std::vector<muGrid::Real> same_values{1.5, 2.5, 3.5};
    void * ptr = same_values.data();
    // equal_value should recognize that the content is the same
    bool is_equal = att.equal_value(ptr);
    BOOST_CHECK(is_equal);

    // Create non-matching void pointer - different final value
    std::vector<muGrid::Real> diff_values{1.5, 2.5, 4.0};
    void * diff_ptr = diff_values.data();
    // equal_value should recognize that the content is different
    bool is_not_equal = att.equal_value(diff_ptr);
    BOOST_CHECK(!is_not_equal);
  }

  BOOST_AUTO_TEST_CASE(NetCDFDimensionStartValueComputation) {
    // Test 1.1: Dimension layout helper methods
    const muGrid::FieldCollection::SubPtMap_t & nb_sub_pts{{"one", 1}};

    const DynGridIndex & nb_domain_grid_pts{3, 3};
    const DynGridIndex & nb_subdomain_grid_pts{3, 3};
    const DynGridIndex & subdomain_locations{0, 0};
    muGrid::GlobalFieldCollection global_fc(nb_domain_grid_pts,
                                            nb_subdomain_grid_pts,
                                            subdomain_locations, nb_sub_pts);

    const std::string & unique_name{"test_field"};
    global_fc.register_real_field(unique_name, 1);

    muGrid::Field & field{global_fc.get_field(unique_name)};

    NetCDFDimensions dimensions;
    std::vector<std::shared_ptr<NetCDFDim>> dim_sp;
    dimensions.add_field_dims_global(field, dim_sp);

    // Create a NetCDFVar to test dimension computation
    NetCDFVariables variables;
    NetCDFVarBase & var{variables.add_field_var(field, dim_sp)};

    // Test that get_start_global() successfully computes start values
    // using the extracted helper methods. Verify it returns non-empty vector
    auto starts = var.get_start_global(0);
    BOOST_CHECK(!starts.empty());
    // Verify that all start values are numeric (can be accessed)
    for (size_t i = 0; i < starts.size(); ++i) {
      BOOST_CHECK(starts[i] >= 0);  // Start indices should be non-negative
    }
  }

  BOOST_AUTO_TEST_CASE(NetCDFAttributeTypeDetection) {
    // Test that attribute types are correctly detected from constructors
    // This ensures variant assignment is type-safe

    // Implicit type detection
    std::vector<char> c{'x'};
    muGrid::NetCDFAtt att_c("c", c);
    BOOST_CHECK_EQUAL(att_c.get_data_type(), muGrid::MU_NC_CHAR);

    std::vector<muGrid::Int> i{42};
    muGrid::NetCDFAtt att_i("i", i);
    BOOST_CHECK_EQUAL(att_i.get_data_type(), muGrid::MU_NC_INT);

    std::vector<muGrid::Uint> u{99};
    muGrid::NetCDFAtt att_u("u", u);
    BOOST_CHECK_EQUAL(att_u.get_data_type(), muGrid::MU_NC_UINT);

    std::vector<muGrid::Index_t> l{999};
    muGrid::NetCDFAtt att_l("l", l);
    BOOST_CHECK_EQUAL(att_l.get_data_type(), muGrid::MU_NC_INDEX_T);

    std::vector<muGrid::Real> d{3.14};
    muGrid::NetCDFAtt att_d("d", d);
    BOOST_CHECK_EQUAL(att_d.get_data_type(), muGrid::MU_NC_REAL);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
