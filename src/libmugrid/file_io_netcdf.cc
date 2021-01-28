/**
 * @file   file_io_netcdf.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   25 Mai 2020
 *
 * @brief  Using the FileIOBase class to implement a serial and parallel I/O
 *         interface for NetCDF files
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

#include "exception.hh"
#include "iterators.hh"
#include "units.hh"

#include "file_io_netcdf.hh"

using muGrid::operator<<;

namespace muGrid {

  FileIONetCDF::FileIONetCDF(const std::string & file_name,
                             const FileIOBase::OpenMode & open_mode,
                             Communicator comm)
      : FileIOBase(file_name, open_mode, comm), dimensions(),
        variables(), nb_sub_pts{{this->pixel, 1}},
        GFC_local_pixels(muGrid::Unknown, nb_sub_pts) {
    this->open();
  }

  /* ---------------------------------------------------------------------- */
  FileIONetCDF::~FileIONetCDF() {
    if (this->netcdf_id != -1) {
      this->close();
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_field_collection(
      muGrid::FieldCollection & fc, std::vector<std::string> field_names,
      std::vector<std::string> state_field_unique_prefixes) {
    // By default register all fields of the field collection if field_names is
    // the default, field_names == REGISTER_ALL_FIELDS. An empty vector is
    // preserved to register no fields. This can be usefull if you e.g. only
    // want to register state fields
    if (field_names.size() == 1) {
      if (field_names[0] == std::string(REGISTER_ALL_FIELDS)) {
        field_names = fc.list_fields();
      }
    }

    // By default register all state fields of the field collection if
    // state_field_unique_prefixes is the default, state_field_unique_prefixes
    // == REGISTER_ALL_STATE_FIELDS. An empty vector is preserved to register no
    // state fields. This can be usefull if you e.g. only want to register
    // fields
    if (state_field_unique_prefixes.size() == 1) {
      if (state_field_unique_prefixes[0] ==
          std::string(REGISTER_ALL_STATE_FIELDS)) {
        state_field_unique_prefixes = fc.list_state_field_unique_prefixes();
      }
    }

    /* register all field collections */
    muGrid::FieldCollection::ValidityDomain domain{fc.get_domain()};
    switch (domain) {
    case muGrid::FieldCollection::ValidityDomain::Global:
      register_field_collection_global(
          dynamic_cast<muGrid::GlobalFieldCollection &>(fc), field_names,
          state_field_unique_prefixes);
      break;
    case muGrid::FieldCollection::ValidityDomain::Local:
      register_field_collection_local(
          dynamic_cast<muGrid::LocalFieldCollection &>(fc), field_names,
          state_field_unique_prefixes);
      break;
    default:
      throw FileIOError(
          "Your field collection does not belong to a valid domain (either "
          "ValidityDomain::Local or ValidityDomain::Global is possible).");
    }

    // Bring NetCDF file in define mode if it is not, this happens if
    // register_field_collection is called several times.
    if (this->netcdf_mode != NetCDFMode::DefineMode) {
      int status{ncmu_redef(this->netcdf_id)};
      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }
      this->netcdf_mode = NetCDFMode::DefineMode;
    }

    if (this->open_mode == FileIOBase::OpenMode::Write) {
      // define dimensions, variables and attributes if in write mode
      define_netcdf_dimensions(this->dimensions);
      define_netcdf_variables(this->variables);
      define_netcdf_attributes(this->variables);

      /* end definitions: leave define mode (collective) */
      ncmu_enddef(this->netcdf_id);
      this->netcdf_mode =
          NetCDFMode::DataMode;  // leave DefineMode -> enter DataMode

    } else if (this->open_mode == FileIOBase::OpenMode::Read or
               this->open_mode == FileIOBase::OpenMode::Append) {
      // inquire dimensions, variables and attributes if in read or append
      // mode

      // variables to store the amount of dimensions, variables, global
      // attributes and the id of the unlimited dimension if existing.
      int ndims{}, nvars{}, ngatts{}, unlimdimid{};
      int status{
          ncmu_inq(this->netcdf_id, &ndims, &nvars, &ngatts, &unlimdimid)};
      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }

      // register dimension IDs
      register_netcdf_dimension_ids(ndims, unlimdimid);

      // register variable IDs
      register_netcdf_variable_ids(nvars);

      // register attribute names
      register_netcdf_attribute_names();  // there is no unique ID for
                                          // attributes
      // register attribute values
      register_netcdf_attribute_values();  // It is necessary to already read
                                           // in the values because they are
                                           // needed for the local fields to
                                           // find the associated pixels field
    } else {
      throw FileIOError("Unknown open mode!");
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::open() {
    int err{};
#ifdef WITH_MPI
    MPI_Info info{MPI_INFO_NULL};
#endif  // WITH_MPI

    if (this->open_mode == FileIOBase::OpenMode::Write) {
      // write/create a new NetCDF file if it does not already exist.
      int cmode{NC_NOCLOBBER | NC_64BIT_DATA};
#ifdef WITH_MPI
      err = ncmu_create(this->comm.get_mpi_comm(), (this->file_name).c_str(),
                        cmode, info, &this->netcdf_id);
#else   // WITH_MPI
      err = ncmu_create((this->file_name).c_str(), cmode, &this->netcdf_id);
#endif  // WITH_MPI

      if (err == NC_EEXIST) {
        std::string e_message =
            "The file '" + this->file_name +
            "' already exists. Please choose an other open_mode or an other "
            "file name or delete the existing file.";
        throw FileIOError(e_message);
      }
      if (err != NC_NOERR) {
        throw FileIOError(ncmu_strerror(err));
      }

      // add "frame" as UNLIMITED dimension
      std::string dim_name{"frame"};
      IOSize_t dim_size{NC_UNLIMITED};
      this->dimensions.add_dim(dim_name, dim_size);
    } else {  // read/append an already existing NetCDF file.
      // add "frame" as UNLIMITED dimension (This is done explicitly because the
      // dimension frame is not represented in the FieldCollections and thus has
      // to be added manually because it is not registered during registering
      // the FieldCollections)
      std::string dim_name{"frame"};
      IOSize_t dim_size{NC_UNLIMITED};
      this->dimensions.add_dim(dim_name, dim_size);

      int omode{};
      switch (this->open_mode) {
      case FileIOBase::OpenMode::Read: {
        omode = NC_NOWRITE;
        break;
      }
      case FileIOBase::OpenMode::Append: {
        omode = NC_WRITE;
        break;
      }
      default:
        omode = NC_NOWRITE;
      }
#ifdef WITH_MPI
      err = ncmu_open(this->comm.get_mpi_comm(), (this->file_name).c_str(),
                      omode, info, &this->netcdf_id);
#else   // WITH_MPI
      err = ncmu_open((this->file_name).c_str(), omode, &this->netcdf_id);
#endif  // WITH_MPI

      // get the frame number
      int unlimdimid{};
      int status{ncmu_inq_unlimdim(this->netcdf_id, &unlimdimid)};
      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }
      IOSize_t frame_len{};
      if (unlimdimid != -1) {
        int status{ncmu_inq_dimlen(this->netcdf_id, unlimdimid, &frame_len)};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
        this->nb_frames = frame_len;
      }
    }

    if (err != NC_NOERR) {
      throw FileIOError(ncmu_strerror(err));
    }

    // file is either created or opened
    this->netcdf_mode = NetCDFMode::DefineMode;

    // if you run in serial set NetCDF to _NoFill mode to be conform with the
    // PnetCDF default. We decide to use _NoFill because this is performance
    // wise better, because it prevents double writing on the same variable
#ifndef WITH_MPI
    // From:
    // https://www.unidata.ucar.edu/software/netcdf/docs/group__datasets.html#ga610e6fadb14a51f294b322a1b8ac1bec //NOLINT
    // Caution: The use of this feature may not be available (or even needed) in
    // future releases. Programmers are cautioned against heavy reliance upon
    // this feature.
    nc_set_fill(this->netcdf_id, NC_NOFILL, nullptr);
#endif  // not WITH_MPI

    // TODO(RLeute): save all necessary header/meta data in the NetCDF file
    //               e.g. muSpectre version, git hash, date, ...
    //               Do this by global attributes 'NC_GLOBAL'
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::close() {
    int err{ncmu_close(this->netcdf_id)};
    if (err != NC_NOERR) {
      throw FileIOError(ncmu_strerror(err));
    }
    this->netcdf_id = -1;  // set netcdf_id to invalid value.
    this->netcdf_mode =
        NetCDFMode::UndefinedMode;  // set netcdf_mode to invalid value.
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::read(const Index_t & frame,
                          const std::vector<std::string> & field_names) {
    // read all fields with name in field_names
    for (auto & f_name : field_names) {
      NetCDFVarBase & var{variables.get_variable(f_name)};
      if (var.get_validity_domain() ==
          muGrid::FieldCollection::ValidityDomain::Local) {
        // read in local_pixels field before you read the local variable
        std::string local_field_name{var.get_local_field_name()};
        if (std::find(read_local_pixel_fields.begin(),
                      read_local_pixel_fields.end(),
                      local_field_name) == read_local_pixel_fields.end()) {
          variables.get_variable(local_field_name)
              .read(this->netcdf_id, this->nb_frames, this->GFC_local_pixels,
                    0);
          this->read_local_pixel_fields.push_back(local_field_name);
        }
      }
      // read the variable
      var.read(this->netcdf_id, this->nb_frames, this->GFC_local_pixels, frame);
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::read(const Index_t & frame) {
    const std::vector<std::string> field_names{this->variables.get_names()};
    read(frame, field_names);
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::write(const Index_t & frame,
                           const std::vector<std::string> & field_names) {
    for (auto & f_name : field_names) {
      NetCDFVarBase & var{variables.get_variable(f_name)};
      // write/read the pixels field (hidden variable) for a local variable if
      // necessary
      if (var.get_validity_domain() ==
          muGrid::FieldCollection::ValidityDomain::Local) {
        const std::string & local_field_name{var.get_local_field_name()};

        if (this->open_mode == FileIOBase::OpenMode::Write) {
          if (std::find(written_local_pixel_fields.begin(),
                        written_local_pixel_fields.end(),
                        local_field_name) == written_local_pixel_fields.end()) {
            // write the corresponding hidden field
            variables.get_variable(local_field_name)
                .write(this->netcdf_id, this->nb_frames, this->GFC_local_pixels,
                       0);
            this->written_local_pixel_fields.push_back(local_field_name);
          }
        } else if (this->open_mode == FileIOBase::OpenMode::Append) {
          // read in local_pixels field before you read the local variable
          if (std::find(read_local_pixel_fields.begin(),
                        read_local_pixel_fields.end(),
                        local_field_name) == read_local_pixel_fields.end()) {
            NetCDFVarBase & local_field_var{
                variables.get_variable(local_field_name)};
            local_field_var.read(this->netcdf_id, this->nb_frames,
                                 this->GFC_local_pixels, frame);
            this->read_local_pixel_fields.push_back(local_field_name);
          }
        }
      }
      // write the variable
      var.write(this->netcdf_id, this->nb_frames, this->GFC_local_pixels,
                frame);
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::write(const Index_t & frame) {
    const std::vector<std::string> field_names{this->variables.get_names()};
    this->write(frame, field_names);
  }

  /* ---------------------------------------------------------------------- */
  void
  FileIONetCDF::write_no_frame(const std::vector<std::string> & field_names) {
    // check if all variables do not have the unlimited dimension "frame"
    for (auto & f_name : field_names) {
      const NetCDFVarBase & var{variables.get_variable(f_name)};
      std::vector<std::string> dim_names{var.get_netcdf_dim_names()};
      if (std::find(dim_names.begin(), dim_names.end(), "frame") !=
          dim_names.end()) {
        throw FileIOError(
            "You try to write the variable '" + var.get_name() +
            "' which has the dimension 'frame' with the function "
            "write_no_frame() which is only valid for variables which do not "
            "have the unlimited dimension 'frame'.");
      }
    }
    // set frame to an arbitrary value because it is not relevant for a variable
    // with no frame dimension, we use 0.
    Index_t frame{0};

    // write all fields with name in field_names
    for (auto & f_name : field_names) {
      NetCDFVarBase & var{variables.get_variable(f_name)};
      var.write(this->netcdf_id, this->nb_frames, this->GFC_local_pixels,
                frame);
    }
  }

  /* ---------------------------------------------------------------------- */
  Index_t FileIONetCDF::handle_frame(Index_t frame) const {
    return this->handle_frame(frame, this->nb_frames);
  }

  /* ---------------------------------------------------------------------- */
  Index_t FileIONetCDF::handle_frame(Index_t frame, Index_t tot_nb_frames) {
    if (frame < 0) {
      frame = tot_nb_frames + frame;
    }
    if (frame >= tot_nb_frames) {
      if (tot_nb_frames == 0) {
        throw FileIOError(
            "The file seems to have no appended frames because the required "
            "frame (" +
            std::to_string(frame) +
            ") is lager than the total number of frames (" +
            std::to_string(tot_nb_frames) +
            ") of the NetCDFFileIO object. Try to append a frame by calling "
            "'FileIONetCDF::append_frame()' befor calling functions like "
            "'FileIONetCDF::write()'.");
      } else {
        throw FileIOError(
            "You inquery frame '" + std::to_string(frame) +
            "' but the NetCDFFileIO object has only the frames 0.." +
            std::to_string(tot_nb_frames - 1));
      }
    }
    return frame;
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_field_collection_global(
      muGrid::GlobalFieldCollection & fc_global,
      const std::vector<std::string> & field_names,
      const std::vector<std::string> & state_field_unique_prefixes) {
    std::vector<std::string> hidden_var_names{
        this->GFC_local_pixels.list_fields()};

    // register state fields
    for (auto & unique_prefix : state_field_unique_prefixes) {
      // check if a variable with the unique_prefix was already registered
      std::vector<std::string> var_names{variables.get_names()};
      if (std::find(var_names.begin(), var_names.end(), unique_prefix) !=
          var_names.end()) {
        throw FileIOError("A variable with unique_prefix '" + unique_prefix +
                          "' was already registered. Please register "
                          "each variable only ones!");
      }
      StateField & state_field{fc_global.get_state_field(unique_prefix)};

      std::vector<std::shared_ptr<NetCDFDim>>
          field_dims;  // dimensions belonging to the variable representing
                       // the field "field_names"

      // register frame as first dimension
      field_dims.push_back(dimensions.find_dim(
          "frame", NC_UNLIMITED));
      // register history_index as second dim
      field_dims.push_back(dimensions.add_dim(
          "history_index__" + std::to_string(state_field.get_nb_memory() + 1),
          static_cast<IOSize_t>(state_field.get_nb_memory() + 1)));

      // add dimensions (because all fields of the state field are equal in its
      // dimensions the dimensions of the current field are added)
      dimensions.add_field_dims_global(state_field.current(), field_dims,
                                       unique_prefix);

      // add variables
      NetCDFVarBase & var{
          variables.add_state_field_var(state_field, field_dims)};

      // add attributes
      var.add_attribute_unit();

      // add field names from state field fields to state_field_field_names
      for (const Field & field : state_field.set_fields()) {
        this->state_field_field_names.push_back(field.get_name());
      }
    }

    // register fields
    for (auto & unique_name : field_names) {
      // check if a variable with the name unique_name was already registered
      std::vector<std::string> var_names{variables.get_names()};
      if (std::find(var_names.begin(), var_names.end(), unique_name) !=
          var_names.end()) {
        throw FileIOError("A variable with name '" + unique_name +
                          "' was already registered. Please register "
                          "each variable only ones!");
      }

      if (std::find(this->state_field_field_names.begin(),
                    this->state_field_field_names.end(),
                    unique_name) != this->state_field_field_names.end()) {
        // This Field was already registered as StateField so the registration
        // as Field can be skipped.
        continue;  // go on with registration of next Field
      }

      // check if the variable should be a hidden one because the field is out
      // of the GFC_local_pixels
      bool hidden{false};
      if (std::find(hidden_var_names.begin(), hidden_var_names.end(),
                    unique_name) != hidden_var_names.end()) {
        hidden = true;
      }

      std::vector<std::shared_ptr<NetCDFDim>>
          field_dims;  // dimensions belonging to the variable representing
                       // the field "field_names"
      if (!hidden) {
        field_dims.push_back(dimensions.find_dim(
            "frame", NC_UNLIMITED));  // register frame as first dimension for
                                      // all non hidden variables!
      }

      // add dimensions
      dimensions.add_field_dims_global(fc_global.get_field(unique_name),
                                       field_dims);

      // add variables
      NetCDFVarBase & var{variables.add_field_var(
          fc_global.get_field(unique_name), field_dims, hidden)};

      // add attributes
      var.add_attribute_unit();
    }

    if (!initialised_GFC_local_pixels) {
      // if not already initialised, initialise GFC_local_pixels for the case
      // that local field collections will be registered
      initialise_gfc_local_pixels(fc_global);
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_field_collection_local(
      muGrid::LocalFieldCollection & fc_local,
      const std::vector<std::string> & field_names,
      const std::vector<std::string> & state_field_unique_prefixes) {
    // register the global field collection which manages the writing offsets
    // for the local pixels
    std::vector<std::string> field_name{
        register_lfc_to_gfc_local_pixels(fc_local)};
    register_field_collection_global(this->GFC_local_pixels, field_name,
                                     std::vector<std::string>());

    // register state fields
    for (auto & unique_prefix : state_field_unique_prefixes) {
      // check if a variable with the unique_prefix was already registered
      std::vector<std::string> var_names{variables.get_names()};
      if (std::find(var_names.begin(), var_names.end(), unique_prefix) !=
          var_names.end()) {
        throw FileIOError("A variable with unique_prefix '" + unique_prefix +
                          "' was already registered. Please register "
                          "each variable only ones!");
      }
      StateField & state_field{fc_local.get_state_field(unique_prefix)};

      std::vector<std::shared_ptr<NetCDFDim>>
          field_dims;  // dimensions belonging to the variable representing
                       // the field "field_names"

      // register frame as first dimension
      field_dims.push_back(dimensions.find_dim(
          "frame", NC_UNLIMITED));
      // register history_index as second dim
      field_dims.push_back(dimensions.add_dim(
          "history_index__" + std::to_string(state_field.get_nb_memory() + 1),
          static_cast<IOSize_t>(state_field.get_nb_memory() + 1)));

      // add dimensions (because all fields of the state field are equal in its
      // dimensions the dimensions of the current field are added)
      dimensions.add_field_dims_local(state_field.current(), field_dims,
                                      this->comm, unique_prefix);

      // add variables
      NetCDFVarBase & var{
          variables.add_state_field_var(state_field, field_dims)};
      // book keeping of the associated field in GFC_local_pixels
      // register the local field name
      var.register_local_field_name(field_name[0]);

      // add attributes
      var.add_attribute_unit();
      var.add_attribute_local_pixels_field();

      // add field names from state field fields to state_field_field_names
      for (const Field & field : state_field.set_fields()) {
        this->state_field_field_names.push_back(field.get_name());
      }
    }

    // register fields
    for (auto & unique_name : field_names) {
      // check if a variable with the name unique_name was already registered
      std::vector<std::string> var_names{variables.get_names()};
      if (std::find(var_names.begin(), var_names.end(), unique_name) !=
          var_names.end()) {
        throw FileIOError("A variable with name '" + unique_name +
                          "' was already registered. Please register "
                          "each variable only ones!");
      }

      if (std::find(this->state_field_field_names.begin(),
                    this->state_field_field_names.end(),
                    unique_name) != this->state_field_field_names.end()) {
        // This Field was already registered as StateField so the registration
        // as Field can be skipped.
        continue;  // go on with registration of next Field
      }

      std::vector<std::shared_ptr<NetCDFDim>>
          field_dims{};  // dimensions belonging to the variable representing
                         // the field "field_names"
      field_dims.push_back(dimensions.find_dim(
          "frame", NC_UNLIMITED));  // always register frame as first dimension!

      // add dimensions
      dimensions.add_field_dims_local(fc_local.get_field(unique_name),
                                      field_dims, this->comm);

      // add variables
      NetCDFVarBase & var{
          variables.add_field_var(fc_local.get_field(unique_name), field_dims)};
      // book keeping of the associated field in GFC_local_pixels
      // register the local field name
      var.register_local_field_name(field_name[0]);

      // add attributes to variable
      var.add_attribute_unit();
      var.add_attribute_local_pixels_field();
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::initialise_gfc_local_pixels(
      const muGrid::GlobalFieldCollection & fc_global) {
    const DynCcoord_t & nb_domain_grid_pts{fc_global.get_nb_domain_grid_pts()};
    const DynCcoord_t & nb_subdomain_grid_pts{
        fc_global.get_nb_subdomain_grid_pts()};
    const DynCcoord_t & subdomain_locations{
        fc_global.get_subdomain_locations()};

    // init global field collection "local_pixels"
    this->GFC_local_pixels.initialise(nb_domain_grid_pts, nb_subdomain_grid_pts,
                                      subdomain_locations);
    this->initialised_GFC_local_pixels = true;
  }

  /* ---------------------------------------------------------------------- */
  std::string FileIONetCDF::register_lfc_to_gfc_local_pixels(
      muGrid::LocalFieldCollection & fc_local) {
    std::string field_name{
        fc_local.get_name()};
    if (initialised_GFC_local_pixels) {
      const std::string pixel{"pixel"};
      const Dim_t Dim{oneD};
      muGrid::TypedField<muGrid::Int64> & local_pixels{
          GFC_local_pixels.template register_field<muGrid::Int64>(field_name, 1,
                                                                  pixel)};
      // fill the global fc with the default value -1
      local_pixels.eigen_vec().setConstant(GFC_LOCAL_PIXELS_DEFAULT_VALUE);

      muGrid::T1FieldMap<muGrid::Int64, Mapping::Mut, Dim,
                         muGrid::IterUnit::Pixel>
          local_pixels_map{local_pixels};
      Index_t nb_pixels = local_pixels.get_nb_pixels();
      Index_t global_offset = comm.cumulative_sum(nb_pixels) - nb_pixels;

      if (this->open_mode == FileIOBase::OpenMode::Write) {
        const std::vector<Index_t> & pixel_ids_on_proc{
            fc_local.get_pixel_ids()};
        IOSize_t num{static_cast<IOSize_t>(
            pixel_ids_on_proc.size())};  // number of local pixels
        IOSize_t processor_offset_end{comm.cumulative_sum(num)};
        IOSize_t local_offset{processor_offset_end - num};  // offset begin
        for (auto && local_global : akantu::enumerate(pixel_ids_on_proc)) {
          auto && local_pixel_id{std::get<0>(local_global)};
          auto && global_pixel_id{std::get<1>(local_global)};
          muGrid::Uint64 offset{local_offset + local_pixel_id};
          // netcdf offers only int or long long int but not long int
          muGrid::Int64 offset_lli{static_cast<muGrid::Int64>(offset)};
          Eigen::Map<Eigen::Matrix<muGrid::Int64, 1, 1>, 0, Eigen::Stride<0, 0>>
            fill_value(&offset_lli);
          local_pixels_map[global_pixel_id - global_offset] = fill_value;
        }
      }
    } else {
      throw FileIOError(
          "It seems like you have not registered a global field collection "
          "before you try to register a local field collection. You always "
          "have to register at least one global field collection to define the "
          "global domain before you can register local field collections.");
    }
    return field_name;
  }

  /* ---------------------------------------------------------------------- */
  void
  FileIONetCDF::define_netcdf_dimensions(NetCDFDimensions & dimensions) {
    /* define dimensions: from name and length (collective) */
    for (auto & netcdf_dim : dimensions.get_dim_vector()) {
      // define a dimension only if it was not already defined (thus dim_id
      // == -1)
      if (netcdf_dim->get_id() == DEFAULT_NETCDFDIM_ID) {
        auto && status{
            ncmu_def_dim(this->netcdf_id, netcdf_dim->get_name().c_str(),
                         netcdf_dim->get_size(), &netcdf_dim->set_id())};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::define_netcdf_variables(NetCDFVariables & variables) {
    /* define variables: from name, type, ... (collective) */
    for (std::shared_ptr<NetCDFVarBase> netcdf_var :
         variables.set_var_vector()) {
      // define a variable only if it was not already defined (thus dim_id
      // == -1)
      if (netcdf_var->get_id() == DEFAULT_NETCDFVAR_ID) {
        int status{ncmu_def_var(
            this->netcdf_id, netcdf_var->get_name().c_str(),
            netcdf_var->get_data_type(), netcdf_var->get_ndims(),
            &(netcdf_var->get_netcdf_dim_ids()[0]), &netcdf_var->set_id())};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::define_netcdf_attributes(NetCDFVariables & variables) {
    /* define attributes: from name, type, ... (collective) */
    for (const std::shared_ptr<NetCDFVarBase> & netcdf_var :
         variables.get_var_vector()) {
      for (const NetCDFAtt & att : netcdf_var->get_netcdf_atts()) {
        int status{ncmu_put_att(this->netcdf_id, netcdf_var->get_id(),
                                att.get_name().c_str(), att.get_data_type(),
                                att.get_nelems(), att.get_value())};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_netcdf_dimension_ids(muGrid::Uint64 ndims,
                                                   Index_t unlimdimid) {
    if (ndims < this->dimensions.get_dim_vector().size()) {
      throw FileIOError(
          "It seems like your registered field collection(s) require more "
          "dimensions than I can find in the given NetCDF file.");
    }

    for (auto & dim : this->dimensions.get_dim_vector()) {
      // define a dimension only if it was not already defined (thus dim_id
      // == -1)
      if (dim->get_id() == DEFAULT_NETCDFDIM_ID) {
        int dim_id{};
        int status{
            ncmu_inq_dimid(this->netcdf_id, dim->get_name().data(), &dim_id)};
        if (status != NC_NOERR) {
          if (status == NC_EBADDIM) {
            std::cout << "Hint: Do you maybe try to read a variable or "
                         "dimension which does not exist in the file?"
                      << std::endl;
          }
          throw FileIOError(ncmu_strerror(status));
        }
        dim->register_id(dim_id);
        if (dim_id == unlimdimid) {
          dim->register_unlimited_dim_size();
          // set nb_frames to number of frames in file
          IOSize_t dim_len{};
          int status{ncmu_inq_dimlen(this->netcdf_id, dim->get_id(), &dim_len)};
          if (status != NC_NOERR) {
            throw FileIOError(ncmu_strerror(status));
          }
          this->nb_frames = dim_len;
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_netcdf_variable_ids(muGrid::Uint64 nvars) {
    if (nvars < this->variables.get_var_vector().size()) {
      throw FileIOError(
          "It seems like your registered field collection(s) require more "
          "variables than I can find in the given NetCDF file.");
    }
    for (std::shared_ptr<NetCDFVarBase> var :
         this->variables.set_var_vector()) {
      // define a variable only if it was not already defined (thus dim_id
      // == -1)
      if (var->get_id() == DEFAULT_NETCDFVAR_ID) {
        int var_id{};
        int status{
            ncmu_inq_varid(this->netcdf_id, var->get_name().data(), &var_id)};
        if (status != NC_NOERR) {
          if (status == NC_ENOTVAR) {
            std::cout << "Hint: Do you maybe try to read a variable which does "
                         "not exist in the file?"
                      << std::endl;
          }
          throw FileIOError(ncmu_strerror(status));
        }
        var->register_id(var_id);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_netcdf_attribute_names() {
    for (std::shared_ptr<NetCDFVarBase> var :
         this->variables.set_var_vector()) {
      // inquiry attribute names until you reach the last (status == NC_ENOTATT)
      for (int att_num = 0; att_num < MAX_NB_ATTRIBUTES; att_num++) {
        // find attribute name
        char name[MAX_LEN_ATTRIBUTE_NAME];
        int status_1{ncmu_inq_attname(this->netcdf_id, var->get_id(), att_num,
                                      &name[0])};
        if (status_1 == NC_ENOTATT) {
          break;  // you reached the last attribute of the variable
        }
        if (status_1 != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status_1));
        }
        nc_type att_data_type{};
        IOSize_t att_nelems{};
        int status_2{ncmu_inq_att(this->netcdf_id, var->get_id(), &name[0],
                                  &att_data_type, &att_nelems)};
        if (status_2 != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status_2));
        }
        std::string att_name(&name[0]);
        var->register_attribute(att_name, att_data_type, att_nelems);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FileIONetCDF::register_netcdf_attribute_values() {
    for (std::shared_ptr<NetCDFVarBase> var :
         this->variables.set_var_vector()) {
      for (NetCDFAtt & att : var->set_netcdf_atts()) {
        const std::string & name{att.get_name()};
        // create a void * to a location with enough space to store the
        // returned value
        void * value{att.reserve_value_space()};
        int status{
            ncmu_get_att(this->netcdf_id, var->get_id(), name.c_str(), value)};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
        if (att.is_value_initialised()) {
          // the attribute has already a value and it is checked if it fitts
          // with the read value from the NetCDF file
          bool equal{att.equal_value(value)};
          if (!equal) {
            throw FileIOError(
                "It seems like the registered attribute value originating from "
                "the registered field collection is not equal to the value "
                "read from the netcdf file.\nvariable name: " +
                var->get_name() + "\nattribute name: " + name +
                "\nattribute value from field collection: " +
                att.get_value_as_string() +
                "\nattribute value from NetCDF file:      " +
                att.convert_void_value_to_string(value));
          }
        } else {
          // value was up to now not registered and is registered now
          att.register_value(value);
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<NetCDFDim>
  NetCDFDimensions::add_dim(const std::string & dim_name,
                            const IOSize_t & dim_size) {
    // check if dimension already exists if not create a new NetCDFDim.
    for (auto & netcdf_dim : this->dim_vector) {
      if (netcdf_dim->equal(dim_name, dim_size)) {
        return netcdf_dim;
      }
    }
    NetCDFDim netcdf_dim(dim_name, dim_size);
    this->dim_vector.push_back(std::make_shared<NetCDFDim>(netcdf_dim));
    return this->dim_vector.back();
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFDimensions::add_field_dims_global(
      const muGrid::Field & field,
      std::vector<std::shared_ptr<NetCDFDim>> & field_dims,
      std::string state_field_name) {
    auto & nb_grid_pts{
        dynamic_cast<muGrid::GlobalFieldCollection &>(field.get_collection())
            .get_nb_domain_grid_pts()};                  // x, y, z
    auto & nb_subpts{field.get_nb_sub_pts()};            // s
    auto & nb_dof_per_subpt{field.get_nb_components()};  // n
    std::vector<std::string> grid_names{"nx", "ny", "nz"};

    // add the field dims in the correct order as you want to have in the NetCDF
    // file (frame, subpt_dofs, subpts, nx, ny, nz)
    // the frame was already added before
    if (nb_dof_per_subpt != 1) {
      // for state fields the field name is corrected by using the state field
      // name
      std::string field_name{field.get_name()};
      if (state_field_name.size() != 0) {
        field_name = state_field_name;
      }
      field_dims.push_back(this->add_dim(
          NetCDFDim::compute_dim_name("subpt_dofs", field_name),
          nb_dof_per_subpt));
    }

    if (nb_subpts != 1 or nb_dof_per_subpt != 1) {
      std::string suffix{field.get_sub_division_tag() + "-" +
                         std::to_string(nb_subpts)};
      field_dims.push_back(this->add_dim(
          NetCDFDim::compute_dim_name("subpts", suffix), nb_subpts));
    }

    int i{0};
    for (auto it = nb_grid_pts.begin(); it != nb_grid_pts.end(); ++it) {
      std::string suffix{};
      Index_t & old_n_grid{this->global_domain_grid[i]};
      if (old_n_grid == 0) {
        this->global_domain_grid[i] = *it;
      } else {
        if (old_n_grid != *it) {
          throw FileIOError(
              "You have already registered a global field_collection with " +
              std::to_string(old_n_grid) + " " + grid_names[i] +
              " points. It is only allowed to register global field "
              "collections with the same domain grid in all spatial "
              "directions.");
        }
      }

      field_dims.push_back(this->add_dim(
          NetCDFDim::compute_dim_name(grid_names[i], suffix), *it));
      i++;
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFDimensions::add_field_dims_local(
      const muGrid::Field & field,
      std::vector<std::shared_ptr<NetCDFDim>> & field_dims,
      const Communicator & comm, std::string state_field_name) {
    IOSize_t nb_pts{static_cast<IOSize_t>(field.get_nb_pixels())};  // i
    IOSize_t nb_pts_glob{comm.sum(nb_pts)};  // sum of all local nb_pts
    IOSize_t nb_subpts{static_cast<IOSize_t>(field.get_nb_sub_pts())};  // s
    IOSize_t nb_dof_per_subpt{
        static_cast<IOSize_t>(field.get_nb_components())};  // n

    // add the field dims in the correct order as you want to have in the NetCDF
    // file (frame, subpt_dofs, subpts, pts)
    // the frame was already added before
    if (nb_dof_per_subpt != 1) {
      // for state fields the field name is corrected by using the state field
      // name
      std::string field_name{field.get_name()};
      if (state_field_name.size() != 0) {
        field_name = state_field_name;
      }
      field_dims.push_back(this->add_dim(
          NetCDFDim::compute_dim_name("subpt_dofs", field_name),
          nb_dof_per_subpt));
    }
    if (nb_subpts != 1 or nb_dof_per_subpt != 1) {
      std::string suffix{field.get_sub_division_tag() + "-" +
                         std::to_string(nb_subpts)};
      field_dims.push_back(this->add_dim(
          NetCDFDim::compute_dim_name("subpts", suffix), nb_subpts));
    }
    if (nb_pts_glob != 0) {
      // for state fields the field name is corrected by using the state field
      // name
      std::string field_name{field.get_name()};
      if (state_field_name.size() != 0) {
        field_name = state_field_name;
      }
      field_dims.push_back(
          this->add_dim(NetCDFDim::compute_dim_name("pts", field_name),
                        nb_pts_glob));
    }
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<NetCDFDim>
  NetCDFDimensions::find_dim(const std::string & dim_name,
                             const IOSize_t & dim_size) {
    for (auto & dim : this->dim_vector) {
      if (dim->equal(dim_name, dim_size)) {
        return dim;
      }
    }
    throw FileIOError("The dimension with name '" + dim_name + "' and size '" +
                      std::to_string(dim_size) + "' was not found.");
    return *(this->dim_vector.end());
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<NetCDFDim>
  NetCDFDimensions::find_dim(const std::string & dim_name) {
    for (auto & dim : this->dim_vector) {
      if (dim->get_name() == dim_name) {
        return dim;
      }
    }
    throw FileIOError("The dimension with name '" + dim_name +
                      "' was not found.");
    return *(this->dim_vector.end());
  }

  /* ---------------------------------------------------------------------- */
  const std::vector<std::shared_ptr<NetCDFDim>> &
  NetCDFDimensions::get_dim_vector() const {
    return this->dim_vector;
  }

  /* ---------------------------------------------------------------------- */
  NetCDFDim::NetCDFDim(const std::string & dim_base_name,
                       const IOSize_t & dim_size)
      : size{dim_size}, name{dim_base_name}, initialised{true} {}

  /* ---------------------------------------------------------------------- */
  const int & NetCDFDim::get_id() const { return this->id; }

  /* ---------------------------------------------------------------------- */
  int & NetCDFDim::set_id() { return this->id; }

  /* ---------------------------------------------------------------------- */
  const IOSize_t & NetCDFDim::get_size() const {
    if (!this->initialised) {
      throw FileIOError(
          "The dimension size is " + std::to_string(this->size) +
          ". Probably you have not initialized the size of dimension " +
          this->name);
    }
    return this->size;
  }

  /* ---------------------------------------------------------------------- */
  const std::string & NetCDFDim::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  std::string NetCDFDim::get_base_name() const {
    return compute_base_name(this->name);
  }

  /* ---------------------------------------------------------------------- */
  std::string NetCDFDim::compute_base_name(const std::string & full_name) {
    std::string::size_type start{0};
    std::string::size_type count{full_name.size()};
    // get rid of suffix
    if (full_name.rfind("__") != std::string::npos) {
      count = full_name.rfind("__");
    }
    std::string base_name{full_name.substr(start, count)};
    return base_name;
  }

  /* ---------------------------------------------------------------------- */
  std::string NetCDFDim::compute_dim_name(const std::string & dim_base_name,
                                          const std::string & suffix) {
    std::string dim_name{};
    if (suffix.size() == 0) {
      dim_name = dim_base_name;
    } else {
      dim_name = dim_base_name + "__" + suffix;
    }
    return dim_name;
  }

  /* ---------------------------------------------------------------------- */
  bool NetCDFDim::equal(const std::string & dim_name,
                        const IOSize_t & dim_size) const {
    bool equal{false};
    if (this->name == dim_name) {
      equal = (this->size == dim_size);
    }
    return equal;
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFDim::register_id(const int dim_id) {
    if (this->id != DEFAULT_NETCDFDIM_ID) {
      throw FileIOError("The dimension id " + std::to_string(this->id) +
                        " was already set. You are only allowed to "
                        "register unregistered dimension ids.");
    } else {
      this->id = dim_id;
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFDim::register_unlimited_dim_size() {
    if (this->name == "frame") {
      this->size = NC_UNLIMITED;
    } else {
      throw FileIOError("The only allowed name for the unlimited dimension "
                        "is 'frame'. You try to register the dimension '" +
                        this->name +
                        "' as unlimited dimension which is not possible.");
    }
  }

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<char> & value)
      : att_name{att_name}, data_type{NC_CHAR}, nelems{static_cast<IOSize_t>(
                                                    value.size())},
        value_c{value}, name_initialised{true}, value_initialised{true} {}

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name, const std::string & value)
      : att_name{att_name}, data_type{NC_CHAR}, nelems{static_cast<IOSize_t>(
                                                    value.size())},
        value_c{}, name_initialised{true}, value_initialised{false} {
    char * tmp_char{const_cast<char *>(value.c_str())};
    this->register_value(reinterpret_cast<void *>(tmp_char));
  }

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<muGrid::Int16> & value)
      : att_name{att_name}, data_type{NC_SHORT}, nelems{static_cast<IOSize_t>(
                                                     value.size())},
        value_si{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<int> & value)
      : att_name{att_name}, data_type{NC_INT}, nelems{static_cast<IOSize_t>(
                                                   value.size())},
        value_i{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<float> & value)
      : att_name{att_name}, data_type{NC_FLOAT}, nelems{static_cast<IOSize_t>(
                                                     value.size())},
        value_f{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<double> & value)
      : att_name{att_name}, data_type{NC_DOUBLE}, nelems{static_cast<IOSize_t>(
                                                      value.size())},
        value_d{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<muGrid::Uint16> & value)
      : att_name{att_name}, data_type{NC_USHORT}, nelems{static_cast<IOSize_t>(
                                                      value.size())},
        value_usi{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<unsigned int> & value)
      : att_name{att_name}, data_type{NC_UINT}, nelems{static_cast<IOSize_t>(
                                                    value.size())},
        value_ui{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<muGrid::Int64> & value)
      : att_name{att_name}, data_type{NC_INT64}, nelems{static_cast<IOSize_t>(
                                                     value.size())},
        value_lli{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const std::vector<muGrid::Uint64> & value)
      : att_name{att_name}, data_type{NC_UINT64}, nelems{static_cast<IOSize_t>(
                                                      value.size())},
        value_ulli{value}, name_initialised{true}, value_initialised{true} {};

  /* ---------------------------------------------------------------------- */
  NetCDFAtt::NetCDFAtt(const std::string & att_name,
                       const nc_type & att_data_type,
                       const IOSize_t & att_nelems)
      : att_name{att_name}, data_type{att_data_type}, nelems{att_nelems} {
    if (att_data_type != NC_CHAR && att_data_type != NC_SHORT &&
        att_data_type != NC_INT && att_data_type != NC_FLOAT &&
        att_data_type != NC_DOUBLE && att_data_type != NC_USHORT &&
        att_data_type != NC_UINT && att_data_type != NC_INT64 &&
        att_data_type != NC_UINT64) {
      throw FileIOError(
          "The given attributes data type '" + std::to_string(att_data_type) +
          "' for the attribute with name '" + att_name + "' is not supported.");
    }
    this->name_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  const std::string & NetCDFAtt::get_name() const { return this->att_name; }

  /* ---------------------------------------------------------------------- */
  const nc_type & NetCDFAtt::get_data_type() const { return this->data_type; }

  /* ---------------------------------------------------------------------- */
  const IOSize_t & NetCDFAtt::get_nelems() const { return this->nelems; }

  /* ---------------------------------------------------------------------- */
  const void * NetCDFAtt::get_value() const {
    const void * val{nullptr};
    switch (this->data_type) {
    case NC_CHAR:
      val = value_c.data();
      break;
    case NC_SHORT:
      val = value_si.data();
      break;
    case NC_INT:
      val = value_i.data();
      break;
    case NC_FLOAT:
      val = value_f.data();
      break;
    case NC_DOUBLE:
      val = value_d.data();
      break;
    case NC_USHORT:
      val = value_usi.data();
      break;
    case NC_UINT:
      val = value_ui.data();
      break;
    case NC_INT64:
      val = value_lli.data();
      break;
    case NC_UINT64:
      val = value_ulli.data();
      break;
    default:
      throw FileIOError(
          "Unknown data type of attribute value in 'NetCDFAtt::get_value()'.");
    }
    return val;
  }

  /* ---------------------------------------------------------------------- */
  std::string NetCDFAtt::get_value_as_string() const {
    // Uses muGrid::operator<< to convert the different data types into
    // std::strings
    std::string val{};
    std::ostream & val_os{std::cout};
    std::ostringstream val_ss{};
    switch (this->data_type) {
    case NC_CHAR:
      val_os << value_c;
      break;
    case NC_SHORT:
      val_os << value_si;
      break;
    case NC_INT:
      val_os << value_i;
      break;
    case NC_FLOAT:
      val_os << value_f;
      break;
    case NC_DOUBLE:
      val_os << value_d;
      break;
    case NC_USHORT:
      val_os << value_usi;
      break;
    case NC_UINT:
      val_os << value_ui;
      break;
    case NC_INT64:
      val_os << value_lli;
      break;
    case NC_UINT64:
      val_os << value_ulli;
      break;
    default:
      throw FileIOError("Unknown data type of attribute value in "
                        "'NetCDFAtt::get_value_as_string()'.");
    }
    val_ss << val_os.rdbuf();
    val = val_ss.str();
    return val;
  }

  /* ---------------------------------------------------------------------- */
  std::string NetCDFAtt::convert_void_value_to_string(void * value) const {
    // creates an temporary NETCDFAtt and than uses get_value_as_string(). The
    // parameters nelems and data_type have to be initialised before you can use
    // this function.
    std::string string_value{};
    if (this->name_initialised) {
      std::string tmp_att_name{"temporary_attribute_for_conversion"};
      NetCDFAtt tmp_netcdfatt(tmp_att_name, this->data_type, this->nelems);
      tmp_netcdfatt.register_value(value);
      string_value = tmp_netcdfatt.get_value_as_string();
    } else {
      throw FileIOError(
          "You have to initialise the name data type and number of elements of "
          "the NetCDFAtt object before you can use this function. This is "
          "necessary because the number of elements and the data type is used "
          "for the interpretation of the void pointer.");
    }
    return string_value;
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFAtt::register_value(void * value) {
    switch (this->data_type) {
    case NC_CHAR: {
      char * value_c_ptr{reinterpret_cast<char *>(value)};
      std::vector<char> tmp(value_c_ptr, value_c_ptr + this->nelems);
      this->value_c = tmp;
      break;
    }
    case NC_SHORT: {
      muGrid::Int16 * value_si_ptr{reinterpret_cast<muGrid::Int16 *>(value)};
      std::vector<muGrid::Int16> tmp(value_si_ptr, value_si_ptr + this->nelems);
      this->value_si = tmp;
      break;
    }
    case NC_INT: {
      int * value_i_ptr{reinterpret_cast<int *>(value)};
      std::vector<int> tmp(value_i_ptr, value_i_ptr + this->nelems);
      this->value_i = tmp;
      break;
    }
    case NC_FLOAT: {
      float * value_f_ptr{reinterpret_cast<float *>(value)};
      std::vector<float> tmp(value_f_ptr, value_f_ptr + this->nelems);
      this->value_f = tmp;
      break;
    }
    case NC_DOUBLE: {
      double * value_d_ptr{reinterpret_cast<double *>(value)};
      std::vector<double> tmp(value_d_ptr, value_d_ptr + this->nelems);
      this->value_d = tmp;
      break;
    }
    case NC_USHORT: {
      muGrid::Uint16 * value_usi_ptr{reinterpret_cast<muGrid::Uint16 *>(value)};
      std::vector<muGrid::Uint16> tmp(value_usi_ptr,
                                      value_usi_ptr + this->nelems);
      this->value_usi = tmp;
      break;
    }
    case NC_UINT: {
      unsigned int * value_ui_ptr{reinterpret_cast<unsigned int *>(value)};
      std::vector<unsigned int> tmp(value_ui_ptr, value_ui_ptr + this->nelems);
      this->value_ui = tmp;
      break;
    }
    case NC_INT64: {
      muGrid::Int64 * value_lli_ptr{reinterpret_cast<muGrid::Int64 *>(value)};
      std::vector<muGrid::Int64> tmp(value_lli_ptr,
                                     value_lli_ptr + this->nelems);
      this->value_lli = tmp;
      break;
    }
    case NC_UINT64: {
      muGrid::Uint64 * value_ulli_ptr{
          reinterpret_cast<muGrid::Uint64 *>(value)};
      std::vector<muGrid::Uint64> tmp(value_ulli_ptr,
                                      value_ulli_ptr + this->nelems);
      this->value_ulli = tmp;
      break;
    }
    default:
      throw FileIOError("The registered data type of the attribute '" +
                        this->get_name() +
                        "' is unknown in NetCDFAtt::register_value()");
    }
    this->value_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  void * NetCDFAtt::reserve_value_space() {
    void * value_ptr{nullptr};
    switch (this->data_type) {
    case NC_CHAR: {
      std::vector<char> tmp(this->nelems, 0);
      this->value_c = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_c.data());
      break;
    }
    case NC_SHORT: {
      std::vector<muGrid::Int16> tmp(this->nelems, 0);
      this->value_si = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_si.data());
      break;
    }
    case NC_INT: {
      std::vector<int> tmp(this->nelems, 0);
      this->value_i = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_i.data());
      break;
    }
    case NC_FLOAT: {
      std::vector<float> tmp(this->nelems, 0);
      this->value_f = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_f.data());
      break;
    }
    case NC_DOUBLE: {
      std::vector<double> tmp(this->nelems, 0);
      this->value_d = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_d.data());
      break;
    }
    case NC_USHORT: {
      std::vector<muGrid::Uint16> tmp(this->nelems, 0);
      this->value_usi = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_usi.data());
      break;
    }
    case NC_UINT: {
      std::vector<unsigned int> tmp(this->nelems, 0);
      this->value_ui = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_ui.data());
      break;
    }
    case NC_INT64: {
      std::vector<muGrid::Int64> tmp(this->nelems, 0);
      this->value_lli = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_lli.data());
      break;
    }
    case NC_UINT64: {
      std::vector<muGrid::Uint64> tmp(this->nelems, 0);
      this->value_ulli = tmp;
      value_ptr = reinterpret_cast<void *>(this->value_ulli.data());
      break;
    }
    default:
      throw FileIOError("Unknown data type of attribute value in "
                        "NetCDFAtt::reserve_value_space().");
    }
    return value_ptr;
  }

  bool NetCDFAtt::equal_value(void * value) const {
    bool equal{false};
    switch (this->data_type) {
    case NC_CHAR: {
      char * comp_value{reinterpret_cast<char *>(value)};
      std::vector<char> comp_value_c(comp_value, comp_value + this->nelems);
      equal = (this->value_c == comp_value_c);
      break;
    }
    case NC_SHORT: {
      muGrid::Int16 * comp_value{reinterpret_cast<muGrid::Int16 *>(value)};
      std::vector<muGrid::Int16> comp_value_si(comp_value,
                                               comp_value + this->nelems);
      equal = (this->value_si == comp_value_si);
      break;
    }
    case NC_INT: {
      int * comp_value{reinterpret_cast<int *>(value)};
      std::vector<int> comp_value_i(comp_value, comp_value + this->nelems);
      equal = (this->value_i == comp_value_i);
      break;
    }
    case NC_FLOAT: {
      float * comp_value{reinterpret_cast<float *>(value)};
      std::vector<float> comp_value_f(comp_value, comp_value + this->nelems);
      equal = (this->value_f == comp_value_f);
      break;
    }
    case NC_DOUBLE: {
      double * comp_value{reinterpret_cast<double *>(value)};
      std::vector<double> comp_value_d(comp_value, comp_value + this->nelems);
      equal = (this->value_d == comp_value_d);
      break;
    }
    case NC_USHORT: {
      muGrid::Uint16 * comp_value{reinterpret_cast<muGrid::Uint16 *>(value)};
      std::vector<muGrid::Uint16> comp_value_usi(comp_value,
                                                 comp_value + this->nelems);
      equal = (this->value_usi == comp_value_usi);
      break;
    }
    case NC_UINT: {
      unsigned int * comp_value{reinterpret_cast<unsigned int *>(value)};
      std::vector<unsigned int> comp_value_ui(comp_value,
                                              comp_value + this->nelems);
      equal = (this->value_ui == comp_value_ui);
      break;
    }
    case NC_INT64: {
      muGrid::Int64 * comp_value{reinterpret_cast<muGrid::Int64 *>(value)};
      std::vector<muGrid::Int64> comp_value_lli(comp_value,
                                                comp_value + this->nelems);
      equal = (this->value_lli == comp_value_lli);
      break;
    }
    case NC_UINT64: {
      muGrid::Uint64 * comp_value{reinterpret_cast<muGrid::Uint64 *>(value)};
      std::vector<muGrid::Uint64> comp_value_ulli(comp_value,
                                                  comp_value + this->nelems);
      equal = (this->value_ulli == comp_value_ulli);
      break;
    }
    default:
      throw FileIOError("Unknown data type of the attribute.");
    }
    return equal;
  }

  /* ---------------------------------------------------------------------- */
  bool NetCDFAtt::is_name_initialised() const { return this->name_initialised; }

  /* ---------------------------------------------------------------------- */
  bool NetCDFAtt::is_value_initialised() const {
    return this->value_initialised;
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarBase::NetCDFVarBase(
      const std::string & var_name, const nc_type & var_data_type,
      const IOSize_t & var_ndims,
      const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
      const muGrid::FieldCollection::ValidityDomain & validity_domain,
      bool hidden)
      : name{var_name}, data_type{var_data_type}, ndims{var_ndims},
        netcdf_dims{netcdf_var_dims}, initialised{true},
        validity_domain{validity_domain}, hidden{hidden} {};

  /* ---------------------------------------------------------------------- */
  const std::string & NetCDFVarBase::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  const nc_type & NetCDFVarBase::get_data_type() const {
    return this->data_type;
  }

  /* ---------------------------------------------------------------------- */
  const IOSize_t & NetCDFVarBase::get_ndims() const { return this->ndims; }

  /* ---------------------------------------------------------------------- */
  const int & NetCDFVarBase::get_id() const { return this->id; }

  /* ---------------------------------------------------------------------- */
  int & NetCDFVarBase::set_id() { return this->id; }

  /* ---------------------------------------------------------------------- */
  std::vector<int> NetCDFVarBase::get_netcdf_dim_ids() const {
    std::vector<int> netcdf_dim_ids;
    for (auto & dim : this->netcdf_dims) {
      int dim_id{dim->get_id()};
      netcdf_dim_ids.push_back(dim_id);
    }
    return netcdf_dim_ids;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> NetCDFVarBase::get_netcdf_dim_names() const {
    std::vector<std::string> netcdf_dim_names;
    for (auto & dim : this->netcdf_dims) {
      std::string dim_name{dim->get_name()};
      netcdf_dim_names.push_back(dim_name);
    }
    return netcdf_dim_names;
  }

  /* ---------------------------------------------------------------------- */
  const std::vector<NetCDFAtt> & NetCDFVarBase::get_netcdf_atts() const {
    return this->netcdf_atts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<NetCDFAtt> & NetCDFVarBase::set_netcdf_atts() {
    return this->netcdf_atts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> NetCDFVarBase::get_netcdf_att_names() const {
    std::vector<std::string> netcdf_att_names;
    for (auto & att : this->netcdf_atts) {
      std::string att_name{att.get_name()};
      netcdf_att_names.push_back(att_name);
    }
    return netcdf_att_names;
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::FieldCollection::ValidityDomain &
  NetCDFVarBase::get_validity_domain() const {
    return this->validity_domain;
  }

  /* ---------------------------------------------------------------------- */
  IOSize_t NetCDFVarBase::get_nb_local_pixels() const {
    return this->get_field().get_collection().get_pixel_ids().size();
  }

  /* ---------------------------------------------------------------------- */
  void * NetCDFVarBase::get_buf() const {
    return this->get_field().get_void_data_ptr();
  }

  /* ---------------------------------------------------------------------- */
  IOSize_t NetCDFVarBase::get_bufcount_mpi_global() const {
    Index_t bufcount{this->get_field().get_nb_pixels() *
                     this->get_field().get_nb_sub_pts() *
                     this->get_field().get_nb_components()};
    return static_cast<IOSize_t>(bufcount);
  }

  /* ---------------------------------------------------------------------- */
  IOSize_t NetCDFVarBase::get_bufcount_mpi_local() const {
    Index_t bufcount{this->get_field().get_nb_sub_pts() *
                     this->get_field().get_nb_components()};
    return static_cast<IOSize_t>(bufcount);
  }

  /* ---------------------------------------------------------------------- */
  Datatype_t NetCDFVarBase::get_buftype() const {
#ifdef WITH_MPI
    MPI_Datatype data_type{this->nc_type_to_mpi_datatype(this->data_type)};
    return data_type;
#else   // WITH_MPI
    return this->data_type;
#endif  // WITH_MPI
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t> NetCDFVarBase::get_count_global() const {
    std::vector<IOSize_t> counts{};
    for (auto & dim : this->netcdf_dims) {
      std::string base_name{dim->get_base_name()};
      IOSize_t count{0};

      // find the correct count for each dimension from its base_name
      if (base_name == "frame") {
        count = 1;
      } else if (base_name == "history_index") {
        count = 1;
      } else if (base_name == "nx") {
        if (this->get_field().is_global()) {
          count = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_nb_subdomain_grid_pts()[0]);
        } else {
          throw FileIOError("err_local");
        }
      } else if (base_name == "ny") {
        if (this->get_field().is_global()) {
          count = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_nb_subdomain_grid_pts()[1]);
        } else {
          throw FileIOError("err_local");
        }
      } else if (base_name == "nz") {
        if (this->get_field().is_global()) {
          count = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_nb_subdomain_grid_pts()[2]);
        } else {
          throw FileIOError("err_local");
        }
      } else if (base_name == "subpts") {
        count = static_cast<IOSize_t>(this->get_field().get_nb_sub_pts());
      } else if (base_name == "subpt_dofs") {
        count = static_cast<IOSize_t>(this->get_field().get_nb_components());
      } else {
        throw FileIOError(
            "I can not find the number of indices for the dimension '" +
            dim->get_name() + "' with base_name '" + base_name +
            "'. Probably this case is not implemented.");
      }

      counts.push_back(count);
    }
    return counts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t> NetCDFVarBase::get_count_local() const {
    std::vector<IOSize_t>
        counts{};  // intermediate storage container for counts

    IOSize_t npix{static_cast<IOSize_t>(
        this->get_field().get_nb_pixels())};  // number of pixels on proc
    for (IOSize_t i = 0; i < npix; i++) {
      IOSize_t count{};
      for (auto & dim : this->netcdf_dims) {
        std::string base_name{dim->get_base_name()};
        if (base_name == "frame") {
          count = 1;
        } else if (base_name == "history_index") {
          count = 1;
        } else if (base_name == "pts") {
          count = 1;
        } else if (base_name == "subpts") {
          count = static_cast<IOSize_t>(this->get_field().get_nb_sub_pts());
        } else if (base_name == "subpt_dofs") {
          count = static_cast<IOSize_t>(this->get_field().get_nb_components());
        } else {
          throw FileIOError(
              "I can not find the number of indices for the dimension '" +
              dim->get_name() + "' with base_name '" + base_name +
              "'. Probably this case is not implemented.");
        }

        counts.push_back(count);
      }
    }

    return counts;
  }

  /* ---------------------------------------------------------------------- */
  nc_type NetCDFVarBase::typeid_to_nc_type(const std::type_info & type_id) {
    // possible types:
    // NC_BYTE, NC_CHAR, NC_SHORT, NC_INT, NC_FLOAT and NC_DOUBLE
    // NC_UBYTE, NC_USHORT, NC_UINT, NC_INT64 and NC_UINT64 (for CDF-5)
    nc_type type{NC_NAT};  // Not A Type

    // if (type_id == typeid(byte)) {
    //   type = NC_BYTE;
    // } else
    if (type_id == typeid(char)) {
      type = NC_CHAR;
    } else if (type_id == typeid(muGrid::Int16)) {
      type = NC_SHORT;
    } else if (type_id == typeid(int)) {
      type = NC_INT;
    } else if (type_id == typeid(float)) {
      type = NC_FLOAT;
    } else if (type_id == typeid(double)) {
      type = NC_DOUBLE;
      // } else if  (type_id == typeid(unsigned byte)) {
      //   type = NC_UBYTE;
    } else if (type_id == typeid(muGrid::Uint16)) {
      type = NC_USHORT;
    } else if (type_id == typeid(unsigned int)) {
      type = NC_UINT;
    } else if (type_id == typeid(muGrid::Int64)) {
      type = NC_INT64;
    } else if (type_id == typeid(muGrid::Uint64)) {
      type = NC_UINT64;
    } else {
      std::string name{type_id.name()};
      throw FileIOError("The given type_id '" + name +
                        "' can not be associated with a NetCDF nc_type. "
                        "Probably this case is not implemented in "
                        "NetCDFVarBase::typeid_to_nc_type().");
    }

    return type;
  }

  /* ---------------------------------------------------------------------- */
#ifdef WITH_MPI
  MPI_Datatype
  NetCDFVarBase::nc_type_to_mpi_datatype(const nc_type & data_type) {
    MPI_Datatype mpi_type{MPI_DATATYPE_NULL};
    switch (data_type) {
    case NC_CHAR:  // char
      mpi_type = MPI_CHAR;
      break;
    case NC_SHORT:  // muGrid::Int16
      mpi_type = MPI_SHORT;
      break;
    case NC_INT:  // int
      mpi_type = MPI_INT;
      break;
    case NC_FLOAT:  // float
      mpi_type = MPI_FLOAT;
      break;
    case NC_DOUBLE:  // double
      mpi_type = MPI_DOUBLE;
      break;
    case NC_USHORT:  // muGrid::Uint16
      mpi_type = MPI_UNSIGNED_SHORT;
      break;
    case NC_UINT:  // unsigned int
      mpi_type = MPI_UNSIGNED;
      break;
    case NC_INT64:  // muGrid::Int64
      mpi_type = MPI_LONG_LONG_INT;
      break;
    case NC_UINT64:  // muGrid::Uint64
      mpi_type = MPI_UNSIGNED_LONG_LONG;
      break;
    default:
      throw FileIOError("The given data_type '" + std::to_string(data_type) +
                        "' can not be associated with a MPI_Datatype. "
                        "Probably this case is not implemented.");
    }

    return mpi_type;
  }

#endif  // WITH_MPI

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::register_id(const int var_id) {
    if (this->id != DEFAULT_NETCDFVAR_ID) {
      throw FileIOError(
          "The variable id is " + std::to_string(this->id) +
          "and hence was already set. You are only allowed to register "
          "unregistered variable IDs.");
    } else {
      this->id = var_id;
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::register_local_field_name(
      const std::string & local_field_name) {
    if (this->validity_domain ==
        muGrid::FieldCollection::ValidityDomain::Local) {
      this->local_field_name = local_field_name;
    } else {
      std::string val{};
      std::ostream & val_os{std::cout};
      std::ostringstream val_ss{};
      val_os << this->validity_domain;
      val_ss << val_os.rdbuf();
      val = val_ss.str();
      throw FileIOError("It is only allowed to register a 'local_field_name'  "
                        "for NetCDFVarBases representing a local field. The "
                        "validity_domain of your field is '" +
                        val + "'.");
    }
  }

  /* ---------------------------------------------------------------------- */
  const std::string & NetCDFVarBase::get_local_field_name() const {
    if (this->validity_domain ==
        muGrid::FieldCollection::ValidityDomain::Local) {
      return this->local_field_name;
    } else {
      std::string val{};
      std::ostream & val_os{std::cout};
      std::ostringstream val_ss{};
      val_os << this->validity_domain;
      val_ss << val_os.rdbuf();
      val = val_ss.str();
      throw FileIOError("It is only allowed to inquire the 'local_field_name' "
                        "for NetCDFVarBases representing a local field. The "
                        "validity_domain of your field is '" +
                        val + "'.");
    }
  }

  /* ---------------------------------------------------------------------- */
  bool NetCDFVarBase::get_hidden_status() const { return this->hidden; }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void NetCDFVarBase::add_attribute(const std::string & att_name,
                                    const T & value) {
    NetCDFAtt attribute(att_name, value);
    this->netcdf_atts.push_back(attribute);
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::register_attribute(const std::string & att_name,
                                     const nc_type & att_data_type,
                                     const IOSize_t & att_nelems) {
    // register a attribute only if it is not already registered
    std::vector<std::string> att_names{this->get_netcdf_att_names()};
    if (std::find(att_names.begin(), att_names.end(), att_name) ==
        att_names.end()) {
      NetCDFAtt registered_att(
          att_name, att_data_type,
          att_nelems);  // call constructor without attribute value
      this->netcdf_atts.push_back(registered_att);
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::add_attribute_unit() {
    std::string att_name{"unit"};

    const muGrid::Unit unit{this->get_field().get_physical_unit()};
    std::ostringstream stream("");
    std::ostream & tmp(stream);
    tmp << unit;
    std::string att_val_string{stream.str()};
    std::vector<char> att_val(att_val_string.begin(),
                              att_val_string.end());

    add_attribute(att_name, att_val);
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::add_attribute_local_pixels_field() {
    std::string att_name{"local_pixels_field"};
    std::vector<char> att_val{this->local_field_name.begin(),
                              this->local_field_name.end()};
    add_attribute(att_name, att_val);
  }

  /* ---------------------------------------------------------------------- */
  void *
  NetCDFVarBase::increment_buf_ptr(void * buf_ptr,
                               const IOSize_t & increment_nb_elements) const {
    void * incremented_buf_ptr{nullptr};
    switch (this->data_type) {
    case NC_CHAR:  // NC_CHAR
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<char *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_SHORT:  // NC_SHORT
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<muGrid::Int16 *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_INT:  // NC_INT
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<int *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_FLOAT:  // NC_FLOAT
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<float *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_DOUBLE:  // NC_DOUBLE
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<double *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_USHORT:  // NC_USHORT
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<muGrid::Uint16 *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_UINT:  // NC_UINT
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<unsigned int *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_INT64:  // NC_INT64
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<muGrid::Int64 *>(buf_ptr) + increment_nb_elements);
      break;
    case NC_UINT64:  // NC_UINT64
      incremented_buf_ptr = reinterpret_cast<void *>(
          reinterpret_cast<muGrid::Uint64 *>(buf_ptr) + increment_nb_elements);
      break;
    default:
      throw FileIOError(
          "A pointer increment for the variable data_type '" +
          std::to_string(data_type) +
          "' is not implemented in NetCDFVarBase::increment_buf_ptr().");
    }
    return incremented_buf_ptr;
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::write(const int netcdf_id, const Index_t & tot_nb_frames,
                            GlobalFieldCollection & GFC_local_pixels,
                            const Index_t & frame_index) {
    // check if frame is correct and compute positive frame value
    Index_t frame{FileIONetCDF::handle_frame(frame_index, tot_nb_frames)};

    int status{};
    if (this->get_validity_domain() ==
        muGrid::FieldCollection::ValidityDomain::Global) {
      /**
       * Write global field with ncmu_put_varm_all
       **/
#ifdef WITH_MPI
      status = ncmu_put_varm_all(
          netcdf_id, this->get_id(), this->get_start_global(frame).data(),
          this->get_count_global().data(), this->get_nc_stride().data(),
          this->get_nc_imap_global().data(), this->get_buf(),
          this->get_bufcount_mpi_global(), this->get_buftype());
#else   // WITH_MPI
      status = ncmu_put_varm_all(
          netcdf_id, this->get_id(), this->get_start_global(frame).data(),
          this->get_count_global().data(), this->get_nc_stride().data(),
          this->get_nc_imap_global().data(), this->get_buf());
#endif  // WITH_MPI
      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }
    } else if (this->get_validity_domain() ==
               muGrid::FieldCollection::ValidityDomain::Local) {
      /**
       * Write local field with ncmu_put_varn_all
       **/
      IOSize_t ndims{this->get_ndims()};  // number of dimensions
      std::vector<IOSize_t> starts_vec{this->get_start_local(
          frame, GFC_local_pixels.get_field(this->get_local_field_name()))};
      std::vector<IOSize_t> counts_vec{this->get_count_local()};
      size_t num_requests{
          starts_vec.size() /
          ndims};  // number of subarray requests in ncmu_put_varn_all

      void * buf_ptr{this->get_buf()};
      IOSize_t nb_points{0};
      std::vector<IODiff_t> stride{this->get_nc_stride()};
      std::vector<IODiff_t> imap{this->get_nc_imap_local()};
#ifdef WITH_MPI
      IOSize_t buf_count{this->get_bufcount_mpi_local()};
      Datatype_t buf_type{this->get_buftype()};
      ncmu_begin_indep_data(netcdf_id);
#endif  // WITH_MPI

      for (size_t i = 0; i < num_requests; i++) {
        std::vector<IOSize_t> starts(starts_vec.begin() + i * ndims,
                                     starts_vec.begin() + (i + 1) * ndims);
        std::vector<IOSize_t> counts(counts_vec.begin() + i * ndims,
                                     counts_vec.begin() + (i + 1) * ndims);
#ifdef WITH_MPI
        status = ncmu_put_varm(netcdf_id, this->get_id(), starts.data(),
                               counts.data(), stride.data(), imap.data(),
                               buf_ptr, buf_count, buf_type);
#else   // WITH_MPI
        status =
            ncmu_put_varm(netcdf_id, this->get_id(), starts.data(),
                          counts.data(), stride.data(), imap.data(), buf_ptr);
#endif  // WITH_MPI
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
        // number of written points in loop step
        // TODO(RLeute): nb_points is only correct if the buffer is in Fortran
        // storage order... I have to fix this for at least C storage order or
        // better an arbitrary case
        nb_points = std::accumulate(counts.begin(), counts.end(), 1,
                                    std::multiplies<IOSize_t>());
        buf_ptr = this->increment_buf_ptr(buf_ptr, nb_points);
      }
#ifdef WITH_MPI
      ncmu_end_indep_data(netcdf_id);
#endif  // WITH_MPI
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarBase::read(const int netcdf_id, const Index_t & tot_nb_frames,
                           GlobalFieldCollection & GFC_local_pixels,
                           const Index_t & frame_index) {
    // check if frame is correct and compute positive frame value
    Index_t frame{FileIONetCDF::handle_frame(frame_index, tot_nb_frames)};

    int status{};
    if (this->get_validity_domain() ==
        muGrid::FieldCollection::ValidityDomain::Global) {
      /**
       * Read global field with ncmu_get_varm_all
       **/
#ifdef WITH_MPI
      status = ncmu_get_varm_all(
          netcdf_id, this->get_id(), this->get_start_global(frame).data(),
          this->get_count_global().data(), this->get_nc_stride().data(),
          this->get_nc_imap_global().data(), this->get_buf(),
          this->get_bufcount_mpi_global(), this->get_buftype());
#else   // WITH_MPI
      status = ncmu_get_varm_all(
          netcdf_id, this->get_id(), this->get_start_global(frame).data(),
          this->get_count_global().data(), this->get_nc_stride().data(),
          this->get_nc_imap_global().data(), this->get_buf());
#endif  // WITH_MPI

      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }
    } else if (this->get_validity_domain() ==
               muGrid::FieldCollection::ValidityDomain::Local) {
      /**
       * Read local field with ncmu_get_varn_all
       **/
#ifdef WITH_MPI
      ncmu_begin_indep_data(netcdf_id);
#endif  // WITH_MPI
      IOSize_t ndims{this->get_ndims()};  // number of dimensions
      std::vector<IOSize_t> starts_vec{this->get_start_local(
          frame, GFC_local_pixels.get_field(this->get_local_field_name()))};
      std::vector<IOSize_t> counts_vec{this->get_count_local()};
      size_t num_requests{
          starts_vec.size() /
          ndims};  // number of subarray requests in ncmu_put_varn_all

      void * buf_ptr{this->get_buf()};
      IOSize_t nb_points{0};
      std::vector<IODiff_t> stride{this->get_nc_stride()};
      std::vector<IODiff_t> imap{this->get_nc_imap_local()};
#ifdef WITH_MPI
      IOSize_t buf_count{this->get_bufcount_mpi_local()};
      Datatype_t buf_type{this->get_buftype()};
#endif  // WITH_MPI

      for (size_t i = 0; i < num_requests; i++) {
        std::vector<IOSize_t> starts(starts_vec.begin() + i * ndims,
                                     starts_vec.begin() + (i + 1) * ndims);
        std::vector<IOSize_t> counts(counts_vec.begin() + i * ndims,
                                     counts_vec.begin() + (i + 1) * ndims);
#ifdef WITH_MPI
        status = ncmu_get_varm(netcdf_id, this->get_id(), starts.data(),
                               counts.data(), stride.data(), imap.data(),
                               buf_ptr, buf_count, buf_type);
#else   // WITH_MPI
        status =
            ncmu_get_varm(netcdf_id, this->get_id(), starts.data(),
                          counts.data(), stride.data(), imap.data(), buf_ptr);
#endif  // WITH_MPI
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
        // number of written points in loop step
        // TODO(RLeute): nb_points is only correct if the buffer is in Fortran
        // storage order... I have to fix this for at least C storage order or
        // better an arbitrary case
        nb_points = std::accumulate(counts.begin(), counts.end(), 1,
                                    std::multiplies<IOSize_t>());
        buf_ptr = this->increment_buf_ptr(buf_ptr, nb_points);
      }
#ifdef WITH_MPI
      ncmu_end_indep_data(netcdf_id);
#endif  // WITH_MPI
    }
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarField::NetCDFVarField(
      const std::string & var_name, const nc_type & var_data_type,
      const IOSize_t & var_ndims,
      const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
      muGrid::Field & var_field, bool hidden)
      : NetCDFVarBase(var_name, var_data_type, var_ndims, netcdf_var_dims,
                      var_field.get_collection().get_domain(), hidden),
        field{var_field} {}

  /* ---------------------------------------------------------------------- */
  const muGrid::Field & NetCDFVarField::get_field() const {
    return this->field;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t>
  NetCDFVarField::get_start_global(const Index_t & frame) const {
    if (frame < 0) {
      throw FileIOError("Only positive frame values are allowed in "
                        "'NetCDFVarBase::get_start_global()'. You gave in the "
                        "value frame = " +
                        std::to_string(frame));
    }

    std::vector<IOSize_t> starts{};

    const std::string & err_local{
        "A local grid should not have a grid dimensions in x-, y-, and "
        "z-direction, it should be a flattend array. Therefore no dimension "
        "should be named 'n*_grid'."};

    for (auto & dim : this->netcdf_dims) {
      std::string base_name{dim->get_base_name()};
      IOSize_t start{0};

      // find the correct start for each dimension from its base_name
      if (base_name == "frame") {
        start = static_cast<IOSize_t>(frame);
      } else if (base_name == "nx") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[0]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "ny") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[1]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "nz") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[2]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "subpts") {
        start = 0;
      } else if (base_name == "subpt_dofs") {
        start = 0;
      } else {
        throw FileIOError(
            "I can not find a start offset for the dimension '" +
            dim->get_name() + "' with base_name '" + base_name +
            "'. Probably this case is not implemented.");
      }

      starts.push_back(start);
    }
    return starts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t>
  NetCDFVarField::get_start_local(const Index_t & frame,
                                 muGrid::Field & local_pixels) const {
    if (frame < 0) {
      throw FileIOError(
          "Only positive frame values are allowed in "
          "'NetCDFVarBase::get_start_local()'. You gave in the value frame = " +
          std::to_string(frame));
    }

    std::vector<IOSize_t>
        starts{};  // intermediate storage container for starts
    muGrid::T1FieldMap<muGrid::Int64, Mapping::Mut, 1, muGrid::IterUnit::Pixel>
        local_pixels_map{local_pixels};

    for (auto & val : local_pixels_map) {
      auto & offset{val(0)};
      if (offset != GFC_LOCAL_PIXELS_DEFAULT_VALUE) {
        IOSize_t start{};
        for (auto & dim : this->netcdf_dims) {
          std::string base_name{dim->get_base_name()};
          if (base_name == "frame") {
            start = static_cast<IOSize_t>(frame);
          } else if (base_name == "pts") {
            start = offset;
          } else if (base_name == "subpts") {
            start = 0;
          } else if (base_name == "subpt_dofs") {
            start = 0;
          } else {
            throw FileIOError(
                "I can not find a start offset for the dimension '" +
                dim->get_name() + "' with base_name '" + base_name +
                "'. Probably this case is not implemented.");
          }

          starts.push_back(start);
        }
      }
    }
    return starts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarField::get_nc_stride() const {
    std::vector<IODiff_t> strides{};
    std::vector<Index_t> s{this->get_field().get_pixels_shape()};
    for (auto & dim : this->netcdf_dims) {
      std::string base_name{dim->get_base_name()};
      IODiff_t stride{0};

      // find the correct stride for each dimension from its base_name
      if (base_name == "frame") {
        stride = 1;
      } else if (base_name == "nx") {
        stride = 1;
      } else if (base_name == "ny") {
        stride = 1;
      } else if (base_name == "nz") {
        stride = 1;
      } else if (base_name == "pts") {
        stride = 1;
      } else if (base_name == "subpts") {
        stride = 1;
      } else if (base_name == "subpt_dofs") {
        stride = 1;
      } else {
        throw FileIOError(
            "I can not find the correct stride for the dimension '" +
            dim->get_name() + "' with base_name '" + base_name +
            "'. Probably this case is not implemented.");
      }

      strides.push_back(stride);
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarField::get_nc_imap_global() const {
    // construc imap from field strides
    const IterUnit iter_type{muGrid::IterUnit::SubPt};
    std::vector<IODiff_t> imap_strides{
        this->get_field().get_nb_pixels() *
        this->get_field().get_nb_dof_per_pixel()};  // imap of frame (nb_dofs)
    auto strides_wrong_type{this->get_field().get_strides(iter_type)};
    std::vector<IODiff_t> strides(strides_wrong_type.begin(),
                                  strides_wrong_type.end());
    imap_strides.insert(imap_strides.end(), strides.begin(), strides.end());

    // if frame is not a dimension of the variable I have to cut off the
    // redundant parts
    std::vector<std::string> names{get_netcdf_dim_names()};
    if (std::find(names.begin(), names.end(), "frame") == names.end()) {
      // erase the first N entries
      int N = imap_strides.size() - get_ndims();
      std::vector<decltype(imap_strides)::value_type>(imap_strides.begin() + N,
                                                      imap_strides.end())
          .swap(imap_strides);
    }
    return imap_strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarField::get_nc_imap_local() const {
    // construc imap from field strides
    const IterUnit iter_type{muGrid::IterUnit::SubPt};
    std::vector<IODiff_t> imap_strides{
        this->get_field().get_nb_pixels() *
        this->get_field()
            .get_nb_dof_per_pixel()};  // imap of frame (nb_pix*nb_dofs)
    auto strides_wrong_type{this->get_field().get_strides(iter_type)};
    std::vector<IODiff_t> strides{strides_wrong_type.begin(),
                                  strides_wrong_type.end()};
    imap_strides.insert(imap_strides.end(), strides.begin(), strides.end());

    // if frame is not a dimension of the variable I have to cut off the
    // redundant parts
    std::vector<std::string> names{get_netcdf_dim_names()};
    if (std::find(names.begin(), names.end(), "frame") == names.end()) {
      // erase the first N entries
      size_t N{imap_strides.size() - get_ndims()};
      std::vector<decltype(imap_strides)::value_type>{imap_strides.begin() + N,
                                                      imap_strides.end()}
          .swap(imap_strides);
    }
    return imap_strides;
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarField::write(const int netcdf_id, const Index_t & tot_nb_frames,
                             GlobalFieldCollection & GFC_local_pixels,
                             const Index_t & frame_index) {
    NetCDFVarBase::write(netcdf_id, tot_nb_frames, GFC_local_pixels,
                         frame_index);
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarField::read(const int netcdf_id, const Index_t & tot_nb_frames,
                            GlobalFieldCollection & GFC_local_pixels,
                            const Index_t & frame_index) {
    NetCDFVarBase::read(netcdf_id, tot_nb_frames, GFC_local_pixels,
                        frame_index);
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarStateField::NetCDFVarStateField(
      const std::string & var_name, const nc_type & var_data_type,
      const IOSize_t & var_ndims,
      const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
      muGrid::StateField & var_state_field)
      : NetCDFVarBase(var_name, var_data_type, var_ndims, netcdf_var_dims,
                      var_state_field.get_collection().get_domain(), false),
        state_field{var_state_field} {}

  /* ---------------------------------------------------------------------- */
  const muGrid::Field & NetCDFVarStateField::get_field() const {
    if (this->state_field_index == 0) {
      return this->state_field.current();
    } else {
      return this->state_field.old(this->state_field_index);
    }
  }

  /* ---------------------------------------------------------------------- */
  size_t NetCDFVarStateField::get_nb_fields() const {
    return static_cast<size_t>(this->state_field.get_nb_memory() + 1);
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t>
  NetCDFVarStateField::get_start_global(const Index_t & frame) const {
    if (frame < 0) {
      throw FileIOError("Only positive frame values are allowed in "
                        "'NetCDFVarStateField::get_start_global()'. You gave "
                        "in the value frame = " +
                        std::to_string(frame));
    }

    std::vector<IOSize_t> starts{};

    const std::string & err_local{
        "A local grid should not have a grid dimensions in x-, y-, and "
        "z-direction, it should be a flattend array. Therefore no dimension "
        "should be named 'n*_grid'."};

    for (auto & dim : this->netcdf_dims) {
      std::string base_name{dim->get_base_name()};
      IOSize_t start{0};

      // find the correct start for each dimension from its base_name
      if (base_name == "frame") {
        start = static_cast<IOSize_t>(frame);
      } else if (base_name == "history_index") {
        start = static_cast<IOSize_t>(this->state_field_index);
      } else if (base_name == "nx") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[0]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "ny") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[1]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "nz") {
        if (this->get_field().is_global()) {
          start = static_cast<IOSize_t>(
              dynamic_cast<muGrid::GlobalFieldCollection &>(
                  this->get_field().get_collection())
                  .get_pixels()
                  .get_subdomain_locations()[2]);
        } else {
          throw FileIOError(err_local);
        }
      } else if (base_name == "subpts") {
        start = 0;
      } else if (base_name == "subpt_dofs") {
        start = 0;
      } else {
        throw FileIOError("I can not find a start offset for the dimension '" +
                          dim->get_name() + "' with base_name '" + base_name +
                          "'. Probably this case is not implemented in "
                          "'NetCDFVarStateField::get_start_global()'.");
      }

      starts.push_back(start);
    }
    return starts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IOSize_t>
  NetCDFVarStateField::get_start_local(const Index_t & frame,
                                       muGrid::Field & local_pixels) const {
    if (frame < 0) {
      throw FileIOError("Only positive frame values are allowed in "
                        "'NetCDFVarStateField::get_start_local()'. You gave in "
                        "the value frame = " +
                        std::to_string(frame));
    }

    std::vector<IOSize_t>
        starts{};  // intermediate storage container for starts
    muGrid::T1FieldMap<muGrid::Int64, Mapping::Mut, 1, muGrid::IterUnit::Pixel>
        local_pixels_map{local_pixels};

    for (auto & val : local_pixels_map) {
      auto & offset{val(0)};
      if (offset != GFC_LOCAL_PIXELS_DEFAULT_VALUE) {
        IOSize_t start{};
        for (auto & dim : this->netcdf_dims) {
          std::string base_name{dim->get_base_name()};
          if (base_name == "frame") {
            start = static_cast<IOSize_t>(frame);
          } else if (base_name == "history_index") {
            start = static_cast<IOSize_t>(this->state_field_index);
          } else if (base_name == "pts") {
            start = offset;
          } else if (base_name == "subpts") {
            start = 0;
          } else if (base_name == "subpt_dofs") {
            start = 0;
          } else {
            throw FileIOError(
                "I can not find a start offset for the dimension '" +
                dim->get_name() + "' with base_name '" + base_name +
                "'. Probably this case is not implemented in "
                "'NetCDFVarStateField::get_start_local()'.");
          }

          starts.push_back(start);
        }
      }
    }
    return starts;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarStateField::get_nc_stride() const {
    std::vector<IODiff_t> strides{};
    std::vector<Index_t> s{this->get_field().get_pixels_shape()};
    for (auto & dim : this->netcdf_dims) {
      std::string base_name{dim->get_base_name()};
      IODiff_t stride{0};

      // find the correct stride for each dimension from its base_name
      if (base_name == "frame") {
        stride = 1;
      } else if (base_name == "history_index") {
        stride = static_cast<IODiff_t>(state_field.get_nb_memory() + 1);
      } else if (base_name == "nx") {
        stride = 1;
      } else if (base_name == "ny") {
        stride = 1;
      } else if (base_name == "nz") {
        stride = 1;
      } else if (base_name == "pts") {
        stride = 1;
      } else if (base_name == "subpts") {
        stride = 1;
      } else if (base_name == "subpt_dofs") {
        stride = 1;
      } else {
        throw FileIOError(
            "I can not find the correct stride for the dimension '" +
            dim->get_name() + "' with base_name '" + base_name +
            "'. Probably this case is not implemented in "
            "'NetCDFVarStateField::get_nc_stride()'.");
      }

      strides.push_back(stride);
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarStateField::get_nc_imap_global() const {
    // construc imap from field strides
    const IterUnit iter_type{muGrid::IterUnit::SubPt};
    IODiff_t nb_history{
        static_cast<IODiff_t>(this->state_field.get_nb_memory() + 1)};
    std::vector<IODiff_t> imap_strides{
        this->get_field().get_nb_pixels() *
        this->get_field().get_nb_dof_per_pixel() *
        nb_history};  // imap of frame (nb_dofs * (nb_memory +1))
    imap_strides.push_back(imap_strides[0] /
                           nb_history);  // imap of history (nb_dofs)
    auto strides_wrong_type{this->get_field().get_strides(iter_type)};
    std::vector<IODiff_t> strides(strides_wrong_type.begin(),
                                  strides_wrong_type.end());
    imap_strides.insert(imap_strides.end(), strides.begin(), strides.end());

    // if frame is not a dimension of the variable I have to cut off the
    // redundant parts
    std::vector<std::string> names{get_netcdf_dim_names()};
    if (std::find(names.begin(), names.end(), "frame") == names.end()) {
      // erase the first N entries
      int N = imap_strides.size() - get_ndims();
      std::vector<decltype(imap_strides)::value_type>(imap_strides.begin() + N,
                                                      imap_strides.end())
          .swap(imap_strides);
    }
    return imap_strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<IODiff_t> NetCDFVarStateField::get_nc_imap_local() const {
    // construc imap from field strides
    const IterUnit iter_type{muGrid::IterUnit::SubPt};
    IODiff_t nb_history{
        static_cast<IODiff_t>(this->state_field.get_nb_memory() + 1)};
    std::vector<IODiff_t> imap_strides{
        this->get_field().get_nb_pixels() *
        this->get_field()
            .get_nb_dof_per_pixel()};  // imap of frame
                                       // (nb_pix*nb_dofs*nb_history)
    imap_strides.push_back(imap_strides[0] /
                           nb_history);  // imap of history (nb_pix*nb_dofs)
    auto strides_wrong_type{this->get_field().get_strides(iter_type)};
    std::vector<IODiff_t> strides{strides_wrong_type.begin(),
                                  strides_wrong_type.end()};
    imap_strides.insert(imap_strides.end(), strides.begin(), strides.end());

    // if frame is not a dimension of the variable I have to cut off the
    // redundant parts
    std::vector<std::string> names{get_netcdf_dim_names()};
    if (std::find(names.begin(), names.end(), "frame") == names.end()) {
      // erase the first N entries
      size_t N{imap_strides.size() - get_ndims()};
      std::vector<decltype(imap_strides)::value_type>{imap_strides.begin() + N,
                                                      imap_strides.end()}
          .swap(imap_strides);
    }
    return imap_strides;
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarStateField::write(const int netcdf_id,
                                  const Index_t & tot_nb_frames,
                                  GlobalFieldCollection & GFC_local_pixels,
                                  const Index_t & frame_index) {
    for (size_t new_state_field_index = 0;
         new_state_field_index < this->get_nb_fields();
         new_state_field_index++) {
      // bring the state_field into the right state (set the state_field_index)
      this->state_field_index = new_state_field_index;
      NetCDFVarBase::write(netcdf_id, tot_nb_frames, GFC_local_pixels,
                           frame_index);
    }
  }

  /* ---------------------------------------------------------------------- */
  void NetCDFVarStateField::read(const int netcdf_id,
                                 const Index_t & tot_nb_frames,
                                 GlobalFieldCollection & GFC_local_pixels,
                                 const Index_t & frame_index) {
    for (size_t new_state_field_index = 0;
         new_state_field_index < this->get_nb_fields();
         new_state_field_index++) {
      // bring the state_field into the right state (set the state_field_index)
      this->state_field_index = new_state_field_index;
      NetCDFVarBase::read(netcdf_id, tot_nb_frames, GFC_local_pixels,
                          frame_index);
    }
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVariables & NetCDFVariables::
  operator+=(std::shared_ptr<NetCDFVarBase> & rhs) {
    this->var_vector.push_back(rhs);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarBase & NetCDFVariables::add_field_var(
      muGrid::Field & var_field,
      const std::vector<std::shared_ptr<NetCDFDim>> & var_dims, bool hidden) {
    std::string var_name{var_field.get_name()};
    const std::type_info & type_id{var_field.get_stored_typeid()};
    nc_type var_data_type{NetCDFVarBase::typeid_to_nc_type(type_id)};
    IOSize_t var_ndims{static_cast<IOSize_t>(var_dims.size())};

    this->var_vector.push_back(
        std::make_shared<NetCDFVarField>(var_name, var_data_type, var_ndims,
                                         var_dims, var_field, hidden));

    return *this->var_vector.back();
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarBase & NetCDFVariables::add_state_field_var(
      muGrid::StateField & var_state_field,
      const std::vector<std::shared_ptr<NetCDFDim>> & var_dims) {
    std::string var_name{var_state_field.get_unique_prefix()};
    const std::type_info & type_id{
        var_state_field.current().get_stored_typeid()};
    nc_type var_data_type{NetCDFVarBase::typeid_to_nc_type(type_id)};
    IOSize_t var_ndims{static_cast<IOSize_t>(var_dims.size())};

    this->var_vector.push_back(std::make_shared<NetCDFVarStateField>(
        var_name, var_data_type, var_ndims, var_dims, var_state_field));

    return *this->var_vector.back();
  }

  /* ---------------------------------------------------------------------- */
  const std::vector<std::shared_ptr<NetCDFVarBase>> &
  NetCDFVariables::get_var_vector() const {
    return this->var_vector;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::shared_ptr<NetCDFVarBase>> &
  NetCDFVariables::set_var_vector() {
    return this->var_vector;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> NetCDFVariables::get_names() const {
    std::vector<std::string> names;
    for (auto & var : this->var_vector) {
      if (!var->get_hidden_status()) {
        names.push_back(var->get_name());
      }
    }
    return names;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> NetCDFVariables::get_hidden_names() const {
    std::vector<std::string> names;
    for (auto & var : this->var_vector) {
      if (var->get_hidden_status()) {
        names.push_back(var->get_name());
      }
    }
    return names;
  }

  /* ---------------------------------------------------------------------- */
  const NetCDFVarBase &
  NetCDFVariables::get_variable(const std::string & var_name) const {
    for (auto & var : this->var_vector) {
      if (var->get_name() == var_name) {
        return *var;
      }
    }
    throw FileIOError("The variable with name '" + var_name +
                      "' was not found. Maybe you forgot to register "
                      "the corresponding FieldCollection?");
    return *var_vector.back();
  }

  /* ---------------------------------------------------------------------- */
  NetCDFVarBase & NetCDFVariables::get_variable(const std::string & var_name) {
    for (auto & var : this->var_vector) {
      if (var->get_name() == var_name) {
        return *var;
      }
    }
    throw FileIOError("The variable with name '" + var_name +
                      "' was not found. Maybe you forgot to register "
                      "the corresponding FieldCollection?");
    return *var_vector.back();
  }

}  // namespace muGrid
