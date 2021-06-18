/**
 * @file   file_io_netcdf.hh
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

#ifndef SRC_LIBMUGRID_FILE_IO_NETCDF_HH_
#define SRC_LIBMUGRID_FILE_IO_NETCDF_HH_

#include <string>
#include <memory>
#include <typeinfo>
#include <type_traits>
#include <iomanip>

#include "communicator.hh"
#include "grid_common.hh"
#include "field.hh"
#include "state_field.hh"
#include "field_collection.hh"
#include "field_map_static.hh"
#include "file_io_base.hh"

#ifdef WITH_MPI
#include <pnetcdf.h>
const auto ncmu_create = ncmpi_create;
const auto ncmu_open = ncmpi_open;
const auto ncmu_enddef = ncmpi_enddef;
const auto ncmu_redef = ncmpi_redef;
const auto ncmu_begin_indep_data = ncmpi_begin_indep_data;
const auto ncmu_end_indep_data = ncmpi_end_indep_data;
const auto ncmu_close = ncmpi_close;
const auto ncmu_strerror = ncmpi_strerrno;
const auto ncmu_def_dim = ncmpi_def_dim;
const auto ncmu_def_var = ncmpi_def_var;
const auto ncmu_inq = ncmpi_inq;
const auto ncmu_inq_varid = ncmpi_inq_varid;
const auto ncmu_inq_dimlen = ncmpi_inq_dimlen;
const auto ncmu_inq_dimid = ncmpi_inq_dimid;
const auto ncmu_inq_unlimdim = ncmpi_inq_unlimdim;
const auto ncmu_inq_attname = ncmpi_inq_attname;
const auto ncmu_inq_att = ncmpi_inq_att;
const auto ncmu_get_vara_all = ncmpi_get_vara_all;
const auto ncmu_get_varm_all = ncmpi_get_varm_all;
const auto ncmu_get_varm = ncmpi_get_varm;
const auto ncmu_get_varn_all = ncmpi_get_varn_all;
const auto ncmu_get_att = ncmpi_get_att;
const auto ncmu_put_vara_all = ncmpi_put_vara_all;
const auto ncmu_put_varm_all = ncmpi_put_varm_all;
const auto ncmu_put_varm = ncmpi_put_varm;
const auto ncmu_put_varn_all = ncmpi_put_varn_all;
const auto ncmu_put_att_text = ncmpi_put_att_text;
const auto ncmu_put_att = ncmpi_put_att;
const auto ncmu_rename_att = ncmpi_rename_att;
using Datatype_t = MPI_Datatype;
using IOSize_t = MPI_Offset;
using IODiff_t = MPI_Offset;
#else  // WITH_MPI
#include <netcdf.h>
const auto ncmu_create = nc_create;
const auto ncmu_open = nc_open;
const auto ncmu_enddef = nc_enddef;
const auto ncmu_redef = nc_redef;
const auto ncmu_close = nc_close;
const auto ncmu_strerror = nc_strerror;
const auto ncmu_def_dim = nc_def_dim;
const auto ncmu_def_var = nc_def_var;
const auto ncmu_inq = nc_inq;
const auto ncmu_inq_varid = nc_inq_varid;
const auto ncmu_inq_dimlen = nc_inq_dimlen;
const auto ncmu_inq_dimid = nc_inq_dimid;
const auto ncmu_inq_unlimdim = nc_inq_unlimdim;
const auto ncmu_inq_attname = nc_inq_attname;
const auto ncmu_inq_att = nc_inq_att;
const auto ncmu_get_vara_all = nc_get_vara;
const auto ncmu_get_varm_all = nc_get_varm;
const auto ncmu_get_varm = nc_get_varm;
const auto ncmu_get_varn_all = nc_get_vara;
const auto ncmu_get_att = nc_get_att;
const auto ncmu_put_vara_all = nc_put_vara;
const auto ncmu_put_varm_all = nc_put_varm;
const auto ncmu_put_varm = nc_put_varm;
const auto ncmu_put_varn_all = nc_put_vara;
const auto ncmu_put_att_text = nc_put_att_text;
const auto ncmu_put_att = nc_put_att;
const auto ncmu_rename_att = nc_rename_att;
using Datatype_t = nc_type;
using IOSize_t = size_t;
using IODiff_t = ptrdiff_t;
#endif  // WITH_MPI

namespace muGrid {

  // We cannot rely on a simple mapping between C++ types and NetCDF type
  // identifier, e.g. int does not necessarily map to NC_INT. We need to deduce
  // the NetCDF type identifier from the storage size of the respective data
  // type, since the actual type depends on the 64-bit data model (LP64, ILP64,
  // LLP64, etc...) that depends on the hardware architecture and operating
  // system (Linux, macOS, Windows use different 64-bit data models).

  template <int size>
  constexpr nc_type netcdf_signed_type() {
    static_assert(!(size == 1 || size == 2 || size == 4 || size == 8),
                  "Unsupported integer storage size");
    return NC_UINT64;
  }
  template<>
  constexpr  nc_type netcdf_signed_type<1>() {
    return NC_BYTE;
  }
  template<>
  constexpr nc_type netcdf_signed_type<2>() {
    return NC_SHORT;
  }
  template<>
  constexpr  nc_type netcdf_signed_type<4>() {
    return NC_INT;
  }
  template<>
  constexpr nc_type netcdf_signed_type<8>() {
    return NC_INT64;
  }

  template <int size>
  constexpr nc_type netcdf_unsigned_type() {
    static_assert(!(size == 1 || size == 2 || size == 4 || size == 8),
                  "Unsupported unsigned integer storage size");
    return NC_UINT64;
  }
  template<>
  constexpr nc_type netcdf_unsigned_type<1>() {
    return NC_UBYTE;
  }
  template<>
  constexpr nc_type netcdf_unsigned_type<2>() {
    return NC_USHORT;
  }
  template<>
  constexpr nc_type netcdf_unsigned_type<4>() {
    return NC_UINT;
  }
  template<>
  constexpr nc_type netcdf_unsigned_type<8>() {
    return NC_UINT64;
  }

  template <typename T>
  constexpr nc_type netcdf_type() {
    if (std::is_signed<T>()) {
      return netcdf_signed_type<sizeof(T)>();
    } else {
      return netcdf_unsigned_type<sizeof(T)>();
    }
  }
  template <>
  constexpr nc_type netcdf_type<char>() {
    return NC_CHAR;
  }
  template <>
  constexpr nc_type netcdf_type<float>() {
    return NC_FLOAT;
  }
  template <>
  constexpr nc_type netcdf_type<double>() {
    return NC_DOUBLE;
  }

  // These are the equivalents of NC_CHAR, NC_INT, NC_DOUBLE etc. but now
  // correctly determined for the basic muGrid types Int, Uint, Real, ...

  constexpr nc_type MU_NC_CHAR = netcdf_type<char>();
  constexpr nc_type MU_NC_INT = netcdf_type<muGrid::Int>();
  constexpr nc_type MU_NC_UINT = netcdf_type<muGrid::Uint>();
  constexpr nc_type MU_NC_INDEX_T = netcdf_type<muGrid::Index_t>();
  constexpr nc_type MU_NC_REAL = netcdf_type<muGrid::Real>();

  constexpr static std::int64_t GFC_LOCAL_PIXELS_DEFAULT_VALUE{
      -1};  // default value to fill the global field collection which stores
            // the offsets of the pixels from a local field collection. As the
            // offsets are larger or equal than zero a negative value is used
            // as default. Arbitrary we choose -1.
  constexpr static int DEFAULT_NETCDFDIM_ID{
      -1};  // default id value for uninitalised NetCDFDim objects. Valid ids
            // starting from zero.
  constexpr static int DEFAULT_NETCDFVAR_ID{
      -1};  // default id value for uninitalised NetCDFVar objects. Valid ids
            // starting from zero.

  /**
   * Class to store the properties of a single NetCDF dimension
   * (name, id, size, initialised)
   **/
  class NetCDFDim {
   public:
    //! Default constructor
    NetCDFDim() = delete;

    /**
     * Constructor with the dimension name and size
     */
    NetCDFDim(const std::string & dim_base_name, const IOSize_t & dim_size);

    //! Copy constructor
    NetCDFDim(const NetCDFDim & other) = default;

    //! Move constructor
    NetCDFDim(NetCDFDim && other) = delete;

    //! Destructor
    virtual ~NetCDFDim() = default;

    //! Copy assignment operator
    NetCDFDim & operator=(const NetCDFDim & other) = delete;

    //! Move assignment operator
    NetCDFDim & operator=(NetCDFDim && other) = delete;

    //! get_dimension id
    const int & get_id() const;

    //! set_dimension id
    int & set_id();

    //! get_dimension size
    const IOSize_t & get_size() const;

    //! get_dimension name
    const std::string & get_name() const;

    //! get the base name of the dimension
    std::string get_base_name() const;

    //! compute the base name of a given name.
    static std::string compute_base_name(const std::string & full_name);

    static std::string compute_dim_name(const std::string & dim_base_name,
                                        const std::string & suffix);

    // compute the tensor index of a tensor_dim, this is only possible for a
    // NetCDFDim with base name "tensor_dim". For NetCDFDims with an other base
    // name -1 is returned, which in normal cases never should be returned.
    int compute_tensor_dim_index() const;

    //! compare the dimension is equal to the given dim_name and size
    bool equal(const std::string & dim_name, const IOSize_t & dim_size) const;

    //! register dimension id, only possible if the id was not already
    //! registered (initialised == false).
    void register_id(const int dim_id);

    //! register unlimited dimension size to NC_UNLIMITED this is only possible
    //! for the dimension with name "frame".
    void register_unlimited_dim_size();

   protected:
    int id{DEFAULT_NETCDFDIM_ID};  //!< location to store the returned dimension
                                   //!< ID.
    IOSize_t
        size{};  //!< Length of dimension; that is, number of values for this
                 //!< dimension as an index to variables that use it. 0 is
                 //!< reserved for the unlimited dimension, NC_UNLIMITED.
    std::string
        name;  //!< Dimension name. Must be a legal netCDF identifier.
    bool initialised{
        false};  //!< bool to check the initialisation status of a dimension
                 //!< only true if size was correct initialised.
  };

  /**
   * Class to store the attributes belonging to a NetCDFVar variable
   * (att_name, data_type, nelems, value, name_initialised, value_initialised)
   **/
  class NetCDFAtt {
    friend class NetCDFGlobalAtt;

   public:
    //! Default constructor
    NetCDFAtt() = delete;

    /**
     * Constructor with the attribute name and its value (char, muGrid::Int,
     * muGrid::Uint, muGrid::real) the values are represented by std::vector<T>
     * of the corresponding type 'T'. The type char has an additional
     * convenience constructor which can take also std::string as input.
     */
    // char value
    NetCDFAtt(const std::string & att_name, const std::vector<char> & value);

    // char value, convenience constructor which can take a std::string value
    NetCDFAtt(const std::string & att_name, const std::string & value);

    // muGrid::Int value
    NetCDFAtt(const std::string & att_name,
              const std::vector<muGrid::Int> & value);

    // muGrid::Uint value
    NetCDFAtt(const std::string & att_name,
              const std::vector<muGrid::Uint> & value);

    // muGrid::Index_t value
    NetCDFAtt(const std::string & att_name,
              const std::vector<muGrid::Index_t> & value);

    // muGrid::Real value
    NetCDFAtt(const std::string & att_name,
              const std::vector<muGrid::Real> & value);

    /**
     * Constructor with the attribute name, data_type and nelems
     */
    NetCDFAtt(const std::string & att_name, const nc_type & att_data_type,
              const IOSize_t & att_nelems);

    //! Copy constructor
    NetCDFAtt(const NetCDFAtt & other) = default;

    //! Move constructor
    NetCDFAtt(NetCDFAtt && other) = delete;

    //! Destructor
    virtual ~NetCDFAtt() = default;

    //! Copy assignment operator
    NetCDFAtt & operator=(const NetCDFAtt & other) = delete;

    //! Move assignment operator
    NetCDFAtt & operator=(NetCDFAtt && other) = delete;

    //! return attribute name
    const std::string & get_name() const;

    //! return nc_type data type of attribute value
    const nc_type & get_data_type() const;

    //! return 'nelems' number of values stored in the attribute value
    const IOSize_t & get_nelems() const;

    //! return the size of the attributes raw data in bytes
    IOSize_t get_data_size() const;

    //! return the size of the attributes name in bytes
    IOSize_t get_name_size() const;

    //! return a const pointer on the attribute value
    const void * get_value() const;

    //! return the attribute values in its 'real type', there is a function for
    //! each type indicated by the suffix of the function name. It is only
    //! allowed to call the function of the correct return type. This can be
    //! satisfyed by using NetCDFAtt::get_data_type(), i.e.
    //! muGrid::MU_NC_CHAR    --  get_typed_value_c()
    //! muGrid::MU_NC_INT     --  get_typed_value_i()
    //! muGrid::MU_NC_UINT    --  get_typed_value_ui()
    //! muGrid::MU_NC_INDEX_T --  get_typed_value_l()
    //! muGrid::MU_NC_REAL    --  get_typed_value_d()
    //! If you call a function which is not matching the data type a FileIOError
    //! will be thrown.
    const std::vector<char> & get_typed_value_c() const;
    const std::vector<muGrid::Int> & get_typed_value_i() const;
    const std::vector<muGrid::Uint> & get_typed_value_ui() const;
    const std::vector<muGrid::Index_t> & get_typed_value_l() const;
    const std::vector<muGrid::Real> & get_typed_value_d() const;

    //! return the attribute value as std::string
    std::string get_value_as_string() const;

    //! converts an input void * value into an std::string under the assumption
    //! that its data type is equal to the attributes data type and also its
    //! number of elements corresponds to the attributes number of elements.
    //! Internal the function get_value_as_string() is used.
    std::string convert_void_value_to_string(void * value) const;

    //! register the attribute value from a void * on the value it is assumed
    //! that the value type is the already registered data_type.
    void register_value(void * value);

    //! reserve enoug space for the value returned from the NetCDF file and
    //! return a pointer on it
    void * reserve_value_space();

    //! compares the input void * value with the stored attributes value and
    //! returns true if they have the same value(s) and otherwise false. The
    //! function compares the actual stored value(s) and not the pointers.
    bool equal_value(void * value) const;

    //! return initialisation status of name, data_type and nelems
    bool is_name_initialised() const;

    //! return initialisation status of the value
    bool is_value_initialised() const;

   private:
    //! update the name and value of the NetCDFAtt while keeping the data_type.
    //! The user of this function is responsible to take care of possible
    //! resulting time consuming extensions of the NetCDF header.
    void update_attribute(const std::string & new_att_name,
                          const nc_type & new_att_data_type,
                          const IOSize_t & new_att_nelems,
                          void * new_att_value);

    //! return a non const pointer on the attribute value
    void * get_value_non_const_ptr();

    std::string att_name;
    nc_type data_type;
    IOSize_t nelems{0};

    // possible values are: char, short, int, float, double, unsigned short,
    // unsigned int, std::int64_t and std::uint64_t
    // Only one of these following vectors can be non zero size. Would be
    // represented by std::variant in C++17.
    std::vector<char> value_c{};
    std::vector<muGrid::Int> value_i{};
    std::vector<muGrid::Uint> value_ui{};
    std::vector<muGrid::Index_t> value_l{};
    std::vector<muGrid::Real> value_d{};

    // flags to see whether the attribute was already initialised or not
    bool name_initialised{
        false};  // true if name, data_type and nelems was set, else false
    bool value_initialised{false};  // true if value was set, else false
  };

  /**
   * Class to represent GLOBAL NetCDF attributes which do not belong to a
   * NetCDFVar
   **/
  class NetCDFGlobalAtt : public NetCDFAtt {
   public:
    //! Default constructor
    NetCDFGlobalAtt() = delete;

    /**
     * Constructor from parent with the attribute name and its value (char,
     * muGrid::Int, muGrid::Uint, muGrid::real) the values are represented by
     * std::vector<T> of the corresponding type 'T'. The type char has an
     * additional convenience constructor which can take also std::string as
     * input.
     */
    // char value
    NetCDFGlobalAtt(const std::string & att_name,
                    const std::vector<char> & value);

    // char value, convenience constructor which can take a std::string value
    NetCDFGlobalAtt(const std::string & att_name, const std::string & value);

    // muGrid::Int value
    NetCDFGlobalAtt(const std::string & att_name,
                    const std::vector<muGrid::Int> & value);

    // muGrid::Uint value
    NetCDFGlobalAtt(const std::string & att_name,
                    const std::vector<muGrid::Uint> & value);

    // muGrid::Index_t value
    NetCDFGlobalAtt(const std::string & att_name,
                    const std::vector<muGrid::Index_t> & value);

    // muGrid::Real value
    NetCDFGlobalAtt(const std::string & att_name,
                    const std::vector<muGrid::Real> & value);

    /**
     * Constructor from parent with the attribute name, data_type and nelems
     */
    NetCDFGlobalAtt(const std::string & att_name, const nc_type & att_data_type,
                    const IOSize_t & att_nelems);

    //! Copy constructor
    NetCDFGlobalAtt(const NetCDFGlobalAtt & other) = default;

    //! Move constructor
    NetCDFGlobalAtt(NetCDFGlobalAtt && other) = delete;

    //! Destructor
    virtual ~NetCDFGlobalAtt() = default;

    //! Copy assignment operator
    NetCDFGlobalAtt & operator=(const NetCDFGlobalAtt & other) = delete;

    //! Move assignment operator
    NetCDFGlobalAtt & operator=(NetCDFGlobalAtt && other) = delete;

    //! check if the global attribute was already written to the file;
    //! true  -- the attribute was already written to the file
    //! false -- the attribute was up to now not written to the file
    bool is_already_written_to_file() { return this->is_written; }

    //! call this function after the NetCDFGlobalAtt was written to the NetCDF
    //! file to set is_written to true.
    void was_written() { this->is_written = true; }

    //! changes the global attributes value to new_value if the type of the
    //! new_value matches the old value and the size of the new_value does not
    //! exceed the size of the old value. Here with old value the current calue
    //! of the global attribute is meant.
    template <class T>
    void update_global_attribute(const std::string & new_att_name,
                                 T new_att_value) {
      // construct a temp NetCDFAtt
      NetCDFAtt tmp_netdcdf_att(new_att_name, new_att_value);

      // check data type
      if (this->get_data_type() != tmp_netdcdf_att.get_data_type()) {
        throw FileIOError(
            "The data types of the new (" +
            std::to_string(this->get_data_type()) + ") and the old (" +
            std::to_string(tmp_netdcdf_att.get_data_type()) +
            ") NetCDFGlobalAtt are not equal which is not allowed!");
      }

      // check name size
      if (this->get_name_size() < tmp_netdcdf_att.get_name_size()) {
        throw FileIOError("The new global attribute name exceeds the old name "
                          "size which is not allowed!");
      }

      // check data size
      if (this->get_data_size() < tmp_netdcdf_att.get_data_size()) {
        throw FileIOError("The new global attribute value data exceeds the old "
                          "value data size which is not allowed!");
      }

      // update value and name
      this->update_attribute(tmp_netdcdf_att.get_name(),
                             tmp_netdcdf_att.get_data_type(),
                             tmp_netdcdf_att.get_nelems(),
                             tmp_netdcdf_att.get_value_non_const_ptr());
    }

   protected:
    bool is_written;  //! true  -- the attribute was already written to the file
                      //! false -- the attribute is yet not written to the file
  };

  /**
   * Base class to store the properties of a single NetCDF variable
   * (name, data_type, ndims, id, netcdf_dims, field, netcdf_atts, initialised,
   *  validity_domain, local_field_name, hidden)
   **/
  class NetCDFVarBase {
   public:
    //! Default constructor
    NetCDFVarBase() = delete;

    /**
     * Constructor with the variable name, data type, variable dimensions and a
     * vector of shared pointers to its associated NetCDFDims.
     */
    NetCDFVarBase(
        const std::string & var_name, const nc_type & var_data_type,
        const IOSize_t & var_ndims,
        const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
        const muGrid::FieldCollection::ValidityDomain & validity_domain,
        bool hidden = false);

    //! Copy constructor
    NetCDFVarBase(const NetCDFVarBase & other) = default;

    //! Move constructor
    NetCDFVarBase(NetCDFVarBase && other) = delete;

    //! Destructor
    virtual ~NetCDFVarBase() = default;

    //! Copy assignment operator
    NetCDFVarBase & operator=(const NetCDFVarBase & other) = delete;

    //! Move assignment operator
    NetCDFVarBase & operator=(NetCDFVarBase && other) = delete;

    //! get the name of the NetCDF variable
    const std::string & get_name() const;

    //! get the data type of the NetCDF variable
    const nc_type & get_data_type() const;

    //! get the number of dimensions of the NetCDF variable
    const IOSize_t & get_ndims() const;

    //! get the unique id of the NetCDF variable
    const int & get_id() const;

    //! get a non const reference to the unique id of the NetCDF variable to set
    //! its value
    int & set_id();

    //! get a vector of all dimension ids of the NetCDF variable
    std::vector<int> get_netcdf_dim_ids() const;

    //! get a vector of all dimension names of the NetCDF variable
    std::vector<std::string> get_netcdf_dim_names() const;

    //! get a reference to the field represented by the NetCDF variable
    virtual const muGrid::Field & get_field() const = 0;

    //! get a const std::vector<NetCDFAtt> & of all attributes belonging to the
    //! NetCDFVarBase
    const std::vector<NetCDFAtt> & get_netcdf_attributes() const;

    //! get a non const std::vector<NetCDFAtt> & of all attributes belonging to
    //! the NetCDFVarBase to set the actual values of the attributes
    std::vector<NetCDFAtt> & set_netcdf_attributes();

    //! get a std::vector<std::string> with the names of all attributes
    std::vector<std::string> get_netcdf_attribute_names() const;

    //! get the FieldCollection::ValidityDomain & of the NetCDFVarBase
    const muGrid::FieldCollection::ValidityDomain & get_validity_domain() const;

    //! get the number of pixels of the local field collection living on the
    //! current processor
    IOSize_t get_nb_local_pixels() const;

    //! get a pointer to the raw data for the NetCDF variable
    void * get_buf() const;

    //! An integer indicates the number of MPI derived data type elements in the
    //! global variable buffer to be written to a file.
    IOSize_t get_bufcount_mpi_global() const;

    //! An integer indicates the number of MPI derived data type elements in the
    //! local variable buffer to be written to a file. (this is the buf count
    //! for a single pixel)
    IOSize_t get_bufcount_mpi_local() const;

    //! A data type that describes the memory layout of the variable buffer
    Datatype_t get_buftype() const;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for contiguous global fields written with ncmu_put_varm_all
    virtual std::vector<IOSize_t>
    get_start_global(const Index_t & frame) const = 0;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for distributed local fields written with ncmu_put_varn_all
    virtual std::vector<IOSize_t>
    get_start_local(const Index_t & frame,
                    muGrid::Field & local_pixels) const = 0;

    //! A vector of IOSize_t values specifying the edge lengths along each
    //! dimension of the block of data values to be written. This function gives
    //! the count for contiguous global fields written with ncmu_put_varm_all
    std::vector<IOSize_t> get_count_global() const;

    //! A vector of IOSize_t values specifying the edge lengths along each
    //! dimension of the block of data values to be written. This function gives
    //! the count for distributed local fields written with ncmu_put_varn_all
    std::vector<IOSize_t> get_count_local(muGrid::Field & local_pixels) const;

    // A vector of Size_t integers that specifies the sampling interval
    // along each dimension of the netCDF variable.
    virtual std::vector<IODiff_t> get_nc_stride() const = 0;

    // A vector of IOSize_t integers that the mapping between the dimensions of
    // a NetCDF variable and the in-memory structure of the internal data array.
    virtual std::vector<IODiff_t> get_nc_imap_global() const = 0;

    virtual std::vector<IODiff_t> get_nc_imap_local() const = 0;

    //! Convert a "std::type_info"  into a NetCDF "nc_type" type.
    static nc_type typeid_to_nc_type(const std::type_info & type_id);

#ifdef WITH_MPI
    //! convert a nc_type data_type into a MPI_Datatype
    static MPI_Datatype nc_type_to_mpi_datatype(const nc_type & data_type);
#endif  // WITH_MPI

    //! register variable id, only possible if the id was not already
    //! registered (id=-1).
    void register_id(const int var_id);

    //! register local_field_name, only possible if the variable belongs to a
    //! local field collection
    void register_local_field_name(const std::string & local_field_name);

    //! get the local_field_name, only possible if the variable belongs to a
    //! local field collection
    const std::string & get_local_field_name() const;

    //! return the status of the variable whether it is a hidden=true netCDFVar
    //! which is only used for book keeping of local fields or a normal
    //! NetCDFVarBase hidden=false
    bool get_hidden_status() const;

    //! add an attribute to the variable by its name and value
    //! the following types are supported:
    //! T                                    nc_type    data_type in NetCDf file
    //! ------------------------------------------------------------------------
    //! std::vector<char>                    NC_CHAR    char *
    //! std::vector<muGrid::Int16>           NC_SHORT   short int *
    //! std::vector<int>                     NC_INT     int *
    //! std::vector<float>                   NC_FLOAT   float *
    //! std::vector<double>                  NC_DOUBLE  double *
    //! std::vector<muGrid::Uint16>          NC_USHORT  unsigned short int *
    //! std::vector<std::int64_t>           NC_INT64   long long int *
    //! std::vector<std::uint64_t>          NC_UINT64  unsigned long long int *
    template <typename T>
    void add_attribute(const std::string & att_name, const T & value) {
      NetCDFAtt attribute(att_name, value);
      this->netcdf_atts.push_back(attribute);
    }

    //! register an attribute by its name (att_name), data type (att_data_type)
    //! and number of elements (att_nelems), afterwards you can read in a value
    //! of type 'void *' from a NetCDF file by e.g. ncmu_get_att()
    void register_attribute(const std::string & att_name,
                            const nc_type & att_data_type,
                            const IOSize_t & att_nelems);

    //! add the unit of the field as attribute to the variable
    void add_attribute_unit();

    //! add the name of the associated local pixels field as attribute to the
    //! variable
    void add_attribute_local_pixels_field();

    //! increments the input buf pointer "buf_ptr" by n elements
    //! "increment_nb_elements" and returns the incremented void pointer to the
    //! new buffer position. This function is made to increment the pointer of
    //! the variables field. Therefore the variables data_type is assumed to be
    //! the data type of the input void * buf_ptr. If your input does not have
    //! the same type the increment will give you wrong results
    void * increment_buf_ptr(void * buf_ptr,
                             const IOSize_t & increment_nb_elements) const;

    //! cross check the properties (start, count, stride and imap, which are
    //! crucial for reading and writing) of the initialised NetCDFVarBase for
    //! consistency.
    void consistency_check_global_var() const;

    void consistency_check_local_var(muGrid::Field & local_pixels) const;

    //! actual call of NetCDF functions to write a single NetCDFVar into the
    //! NetCDF file
    virtual void write(const int netcdf_id, const Index_t & tot_nb_frames,
                       GlobalFieldCollection & GFC_local_pixels,
                       const Index_t & frame_index);

    //! actual call of NetCDF functions to read in the data of a single
    //! NetCDFVar from a NetCDF file
    virtual void read(const int netcdf_id, const Index_t & tot_nb_frames,
                      GlobalFieldCollection & GFC_local_pixels,
                      const Index_t & frame_index);

   protected:
    std::string name;  // Variable name. Must be a legal netCDF identifier.
    nc_type data_type{NC_NAT};  // One of the predefined netCDF external data
                                // types. NAT = Not A Type.
    IOSize_t ndims{};           // Number of dimensions of the variable.
    int id{DEFAULT_NETCDFVAR_ID};  // location to store the returned NetCDF
                                   // variable ID. Variable IDs starting at 0.
    std::vector<std::shared_ptr<NetCDFDim>> netcdf_dims{
        nullptr};  // list of NetCDFDims belonging to the variable.
    std::vector<NetCDFAtt> netcdf_atts{};  // vector of all variable attributes
    bool initialised{
        false};  //!< bool to check the initialisation status of a variable
                 //!< only true if ndims was correct initialised.
    const muGrid::FieldCollection::ValidityDomain
        validity_domain{};  // stores whether a variable belongs to a global or
                            // local field.
    std::string local_field_name{};  // name of the field that is storing the
                                     // local pixels
    bool hidden{false};  // internal book keeping variables for the local fields
                         // are hidden (hidden=true) all other variables are
                         // treated normal (hidden=false)
  };

  /**
   * Class to store the properties of a single NetCDF variable representing a
   * Field
   **/
  class NetCDFVarField final : public NetCDFVarBase {
   public:
    //! Default constructor
    NetCDFVarField() = delete;

    /**
     * Constructor with the variable name, data type, variable dimensions and a
     * vector of shared pointers to its associated NetCDFDims.
     */
    NetCDFVarField(
        const std::string & var_name, const nc_type & var_data_type,
        const IOSize_t & var_ndims,
        const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
        muGrid::Field & var_field, bool hidden = false);

    //! Copy constructor
    NetCDFVarField(const NetCDFVarField & other) = default;

    //! Move constructor
    NetCDFVarField(NetCDFVarField && other) = delete;

    //! Destructor
    virtual ~NetCDFVarField() = default;

    //! Copy assignment operator
    NetCDFVarField & operator=(const NetCDFVarField & other) = delete;

    //! Move assignment operator
    NetCDFVarField & operator=(NetCDFVarField && other) = delete;

    //! get a reference to the field represented by the NetCDF variable
    const muGrid::Field & get_field() const;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for contiguous global fields written with ncmu_put_varm_all
    std::vector<IOSize_t> get_start_global(const Index_t & frame) const;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for distributed local fields written with ncmu_put_varn_all
    std::vector<IOSize_t>
    get_start_local(const Index_t & frame,
                    muGrid::Field & local_pixels) const;

    // A vector of Size_t integers that specifies the sampling interval
    // along each dimension of the netCDF variable.
    std::vector<IODiff_t> get_nc_stride() const;

    // A vector of IOSize_t integers that the mapping between the dimensions of
    // a NetCDF variable and the in-memory structure of the internal data array.
    std::vector<IODiff_t> get_nc_imap_global() const;

    std::vector<IODiff_t> get_nc_imap_local() const;

    //! actual call of NetCDF functions to write a single NetCDFVar into the
    //! NetCDF file
    void write(const int netcdf_id, const Index_t & tot_nb_frames,
               GlobalFieldCollection & GFC_local_pixels,
               const Index_t & frame_index);

    //! actual call of NetCDF functions to read in the data of a single
    //! NetCDFVar from a NetCDF file
    void read(const int netcdf_id, const Index_t & tot_nb_frames,
              GlobalFieldCollection & GFC_local_pixels,
              const Index_t & frame_index);

   protected:
    muGrid::Field &
        field;  // Reference to the field in which the variable data is stored
  };

  /**
   * Class to store the properties of a single NetCDF variable representing a
   * StateField. The class behaves like it represents a single Field of the
   * StateField. The state_filed_index decides which Field is represented. The
   * Fields are always ordered such that state_field_index=0 represents the
   * current Field of the StateField and state_field_index=nb_memory represents
   * the oldest Field of the StateField.
   **/
  class NetCDFVarStateField final : public NetCDFVarBase {
    constexpr static size_t DEFAULT_STATE_FIELD_INDEX{
        0};  // represents the default state of the NetCDFVarStateField, i.e.
             // the current Field
   public:
    //! Default constructor
    NetCDFVarStateField() = delete;

    /**
     * Constructor with the variable name, data type, variable dimensions and a
     * vector of shared pointers to its associated NetCDFDims.
     */
    NetCDFVarStateField(
        const std::string & var_name, const nc_type & var_data_type,
        const IOSize_t & var_ndims,
        const std::vector<std::shared_ptr<NetCDFDim>> & netcdf_var_dims,
        muGrid::StateField & var_state_field);

    //! Copy constructor
    NetCDFVarStateField(const NetCDFVarStateField & other) = default;

    //! Move constructor
    NetCDFVarStateField(NetCDFVarStateField && other) = delete;

    //! Destructor
    virtual ~NetCDFVarStateField() = default;

    //! Copy assignment operator
    NetCDFVarStateField & operator=(const NetCDFVarStateField & other) = delete;

    //! Move assignment operator
    NetCDFVarStateField & operator=(NetCDFVarStateField && other) = delete;

    //! get a reference to the field represented by the NetCDF variable
    const muGrid::Field & get_field() const;

    //! return the number of fields belonging to the state field (nb_memory + 1)
    size_t get_nb_fields() const;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for contiguous global fields written with ncmu_put_varm_all
    std::vector<IOSize_t> get_start_global(const Index_t & frame) const;

    //! A vector of IOSize_t values specifying the index in the variable
    //! where the first of the data values will be written. This function gives
    //! the start for distributed local fields written with ncmu_put_varn_all
    std::vector<IOSize_t>
    get_start_local(const Index_t & frame,
                    muGrid::Field & local_pixels) const;

    // A vector of Size_t integers that specifies the sampling interval
    // along each dimension of the netCDF variable.
    std::vector<IODiff_t> get_nc_stride() const;

    // A vector of IOSize_t integers that the mapping between the dimensions of
    // a NetCDF variable and the in-memory structure of the internal data array.
    std::vector<IODiff_t> get_nc_imap_global() const;

    std::vector<IODiff_t> get_nc_imap_local() const;

    //! actual call of NetCDF functions to write a single NetCDFVar into the
    //! NetCDF file
    void write(const int netcdf_id, const Index_t & tot_nb_frames,
               GlobalFieldCollection & GFC_local_pixels,
               const Index_t & frame_index);

    //! actual call of NetCDF functions to read in the data of a single
    //! NetCDFVar from a NetCDF file
    void read(const int netcdf_id, const Index_t & tot_nb_frames,
              GlobalFieldCollection & GFC_local_pixels,
              const Index_t & frame_index);

   protected:
    muGrid::StateField &
        state_field;  // Reference to the state field in which the single
                      // fields, holding the data, are managed.
    size_t state_field_index{
        DEFAULT_STATE_FIELD_INDEX};  // keeps the state of the StateField;
                                     // default: DEFAULT_STATE_FIELD_INDEX, i.e.
                                     // the current Field of the StateField
  };

  /**
   * Class to store the NetCDF dimensions
   * (dim_vector, global_domain_grid)
   **/
  class NetCDFDimensions {
   public:
    NetCDFDimensions() = default;

    //! Copy constructor
    NetCDFDimensions(const NetCDFDimensions & other) = delete;

    //! Move constructor
    NetCDFDimensions(NetCDFDimensions && other) = delete;

    //! Destructor
    virtual ~NetCDFDimensions() = default;

    //! Copy assignment operator
    NetCDFDimensions & operator=(const NetCDFDimensions & other) = delete;

    //! Move assignment operator
    NetCDFDimensions & operator=(NetCDFDimensions && other) = delete;

    //! Add a Dimension given its base name and size
    //! returns a std::shared_ptr<NetCDFDim> to the added NetCDFDim object
    std::shared_ptr<NetCDFDim> add_dim(const std::string & dim_name,
                                       const IOSize_t & dim_size);

    //! Add all dimensions of a global Field (f, (h,) s, n, x, y, z)
    //! f: frame
    //! h: history index for state fields
    //! x: number of points in x direction
    //! y: number of points in y direction
    //! z: number of points in z direction
    //! s: number of sub points per point (pixel)
    //! n: number of DOFs per sub point
    void
    add_field_dims_global(const muGrid::Field & field,
                          std::vector<std::shared_ptr<NetCDFDim>> & field_dims,
                          std::string state_field_name = std::string{});

    //! Add all dimensions of a local Field (f, (h,) s, n, i)
    //! f: frame
    //! h: history index for state fields
    //! s: number of sub points per point (pixel)
    //! n: number of DOFs per sub point
    //! i: total number of points (pixels) in the local field
    void
    add_field_dims_local(const muGrid::Field & field,
                         std::vector<std::shared_ptr<NetCDFDim>> & field_dims,
                         const Communicator & comm,
                         std::string state_field_name = std::string{});

    //! find dimension by unique dim_name and size
    //! returns a std::shared_ptr<NetCDFDim> to the found dimension, if the
    //! dimension is not found it returns the end of the dim_vector of the
    //! NetCDFDimensions object and throws a muGrid::FielIOError.
    std::shared_ptr<NetCDFDim> find_dim(const std::string & dim_name,
                                        const IOSize_t & dim_size);

    //! find dimension only by the unique dim_name
    //! returns a std::shared_ptr<NetCDFDim> to the found dimension, if the
    //! dimension is not found it returns the end of the dim_vector of the
    //! NetCDFDimensions object and throws a muGrid::FielIOError.
    std::shared_ptr<NetCDFDim> find_dim(const std::string & dim_name);

    //! return a std::vector<std::shared_ptr<NetCDFDim>> & of all NetCDFDims
    //! belonging to the NetCDFDimensions
    const std::vector<std::shared_ptr<NetCDFDim>> & get_dim_vector() const;

   protected:
    std::vector<std::shared_ptr<NetCDFDim>> dim_vector{};
    std::vector<Index_t> global_domain_grid{0, 0, 0};
  };

  /**
   * Class to store the GLOBAL NetCDF attributes
   **/
  class NetCDFGlobalAttributes {
   public:
    NetCDFGlobalAttributes() = default;

    //! Copy constructor
    NetCDFGlobalAttributes(const NetCDFGlobalAttributes & other) = delete;

    //! Move constructor
    NetCDFGlobalAttributes(NetCDFGlobalAttributes && other) = delete;

    //! Destructor
    virtual ~NetCDFGlobalAttributes() = default;

    //! Copy assignment operator
    NetCDFGlobalAttributes &
    operator=(const NetCDFGlobalAttributes & other) = delete;

    //! Move assignment operator
    NetCDFGlobalAttributes &
    operator=(NetCDFGlobalAttributes && other) = delete;

    //! add a global attribute in all available value type versions
    template <class T>
    NetCDFGlobalAtt & add_attribute(const std::string & global_att_name,
                                    const T & value) {
      this->check_global_attribute_name(global_att_name);
      this->global_att_vector.push_back(
          std::make_shared<NetCDFGlobalAtt>(global_att_name, value));

      return *this->global_att_vector.back();
    }

    //! get a global attribute from the global_att_vector by its name
    const NetCDFGlobalAtt &
    get_attribute(const std::string & global_att_name) const;

    //! get a const std::vector<std::shared_ptr<NetCDFGlobalAtt>> of all
    //! NetCDFGlobalAtts belonging to the NetCDFGlobalAttributes object, i.e. a
    //! const view on the global_att_vector.
    const std::vector<std::shared_ptr<NetCDFGlobalAtt>>
    get_global_attribute_vector() const;

    //! get a std::vector<std::shared_ptr<NetCDFGlobalAtt>> of all
    //! NetCDFGlobalAtts belonging to the NetCDFGlobalAttributes object, i.e. a
    //! non const view on the global_att_vector.
    std::vector<std::shared_ptr<NetCDFGlobalAtt>> set_global_attribute_vector();

    //! get a std::shared_ptr<NetCDFGlobalAtt> of the NetCDFGlobalAtt with name
    //! 'global_att_name' from the global_att_vector.
    std::shared_ptr<NetCDFGlobalAtt>
    set_global_attribute(const std::string & global_att_name);

    //! get a std::vector<std::string> of all global attribute names
    std::vector<std::string> get_global_attribute_names() const;

    //! register a global attribute by its name (g_att_name), data type
    //! (g_att_data_type) and number of elements (g_att_nelems), afterwards you
    //! can read in a value of type 'void *' from a NetCDF file by e.g.
    //! ncmu_get_att()
    void register_attribute(const std::string & g_att_name,
                            const nc_type & g_att_data_type,
                            const IOSize_t & g_att_nelems);

    //! std::string todays date
    std::string todays_date() const;

    //! std::string time now
    std::string time_now() const;

    //! add the actual date and time to the global_att_vector as creation_date
    //! and creation_time
    void add_date_and_time(std::string name_prefix = "creation");

    //! add muGrid version information
    void add_muGrid_version_info();

   protected:
    //! check if the global_att_name is valid/already in use
    //! no Error -- if global_att_name is valid and was not used before
    //! Error -- if global_att_name is in use and can not be used twice
    void check_global_attribute_name(const std::string global_att_name);

    std::vector<std::shared_ptr<NetCDFGlobalAtt>> global_att_vector{};
  };

  /**
   * Class to store the NetCDF variables
   **/
  class NetCDFVariables {
   public:
    NetCDFVariables() = default;

    //! Copy constructor
    NetCDFVariables(const NetCDFVariables & other) = delete;

    //! Move constructor
    NetCDFVariables(NetCDFVariables && other) = delete;

    //! Destructor
    virtual ~NetCDFVariables() = default;

    //! Copy assignment operator
    NetCDFVariables & operator=(const NetCDFVariables & other) = delete;

    //! Move assignment operator
    NetCDFVariables & operator=(NetCDFVariables && other) = delete;

    //! Add operator for a single NetCDFVarBase
    NetCDFVariables & operator+=(std::shared_ptr<NetCDFVarBase> & rhs);

    //! Add a local or global field as variable and attach the dimensions to it.
    NetCDFVarBase &
    add_field_var(muGrid::Field & var_field,
                  const std::vector<std::shared_ptr<NetCDFDim>> & var_dims,
                  bool hidden = false);

    //! Add a local or global state field as variable and attach the dimensions
    //! to it.
    NetCDFVarBase & add_state_field_var(
        muGrid::StateField & var_state_field,
        const std::vector<std::shared_ptr<NetCDFDim>> & var_dims);

    //! return a const reference on the var_vector which stores all variables
    const std::vector<std::shared_ptr<NetCDFVarBase>> & get_var_vector() const;

    //! return a non const reference on the var_vector which stores all
    //! variables to modify the NetCDFVarBase objects
    std::vector<std::shared_ptr<NetCDFVarBase>> & set_var_vector();

    //! vector of all variable names (i.e. all field names stored in variables)
    //! with exception of the book keeping variables for the local fields which
    //! can be given by get_hidden_names()
    std::vector<std::string> get_names() const;

    //! vector of all book keeping variables for the registered local fields
    std::vector<std::string> get_hidden_names() const;

    //! get a NetCDFVarBase variable from the var_vector by its unique name
    const NetCDFVarBase & get_variable(const std::string & var_name) const;

    //! get a NetCDFVarBase variable from the var_vector by its unique name
    NetCDFVarBase & get_variable(const std::string & var_name);

   protected:
    std::vector<std::shared_ptr<NetCDFVarBase>> var_vector{};
  };

  /**
   * FileIO class for NetCDF files.
   */
  class FileIONetCDF : public FileIOBase {
   public:
    constexpr static int MAX_NB_ATTRIBUTES{
        10};  // maximum number of allowed attributes per variable
    constexpr static int MAX_LEN_ATTRIBUTE_NAME{
        20};  // maximum length of an attribute name
    constexpr static int MAX_NB_GLOBAL_ATTRIBUTES{
        30};  // maximum number of allowed global attributes per FileIONetCDF
              // object
    constexpr static int MAX_LEN_GLOBAL_ATTRIBUTE_NAME{
        25};  // maximum length of an global attribute name

    enum class NetCDFMode { UndefinedMode, DefineMode, DataMode };
    //! Default constructor
    FileIONetCDF() = delete;

    /**
     * Constructor with the domain's number of grid points in each direciton,
     * the number of components to transform, and the communicator
     */
    FileIONetCDF(const std::string & file_name,
                 const FileIOBase::OpenMode & open_mode,
                 Communicator comm = Communicator());

    //! Copy constructor
    FileIONetCDF(const FileIONetCDF & other) = delete;

    //! Move constructor
    FileIONetCDF(FileIONetCDF && other) = delete;

    //! Destructor
    virtual ~FileIONetCDF();

    //! Copy assignment operator
    FileIONetCDF & operator=(const FileIONetCDF & other) = delete;

    //! Move assignment operator
    FileIONetCDF & operator=(FileIONetCDF && other) = delete;

    //! Tell the I/O object about the field collections we want to dump to this
    //! file before the file is opened
    //! @parameter field_names -- name of fields from the field collection that
    //! schould be saved in the NetCDF file. This parameter should be used if
    //! not all fields from the field collection will be written to the file
    //! (default case).
    void register_field_collection(
        muGrid::FieldCollection & fc,
        std::vector<std::string> field_names = {REGISTER_ALL_FIELDS},
        std::vector<std::string> state_field_unique_prefixes = {
            REGISTER_ALL_STATE_FIELDS}) final;

    //! close file
    void close() final;

    //! read the fields identified by `field_names` frame from file
    void read(const Index_t & frame,
              const std::vector<std::string> & field_names) final;

    //! read the fields in frame from file
    void read(const Index_t & frame) final;

    //! write contents of all fields within the field collection with the name
    //! in field_names to the frame in file
    void write(const Index_t & frame,
               const std::vector<std::string> & field_names) final;

    //! write contents of all fields within the field collection to the file
    void write(const Index_t & frame) final;

    //! checks if the frame is valid and computes the corresponding positive
    //! frame value for a negative frame value. Examples:
    //! a) nb_frames = 5; frame_in = -3; frame_out = 2
    //! b) nb_frames = 5; frame_in = 3; frame_out = 3
    //! c) nb_frames = 5; frame_in = 7; Error
    //! d) nb_frames = 5; frame_in = -6; Error
    Index_t handle_frame(Index_t frame) const;
    //! static version of handle_frame
    static Index_t handle_frame(Index_t frame, Index_t tot_nb_frames);

    //! register a NetCDFGlobalAtt to the global_attributes of FileIONetCDF.
    //! This function can only be used in open_mode =
    //! FileIOBase::OpenMode::Write and before write() was called the first time
    //! otherwise there is the danger of having time expensive NetCDF header
    //! expansions, which is the reason why this is prevented.
    template <class T>
    void write_global_attribute(const std::string & att_name, T value) {
      if (this->global_attributes_defined) {
        throw FileIOError(
            "It is forbidden to write gloabal attributes after you have called "
            "'FileIONetCDF::write()' the first time. This is to prevent "
            "probably time expensive expansions of the NetCDF header after "
            "there was data written to the NetCDF file. Therefore, please "
            "write all global attributes before you write data to your file.");
      }
      if (this->open_mode != FileIOBase::OpenMode::Write) {
        throw FileIOError(
            "It is only possible to write global attributes when the "
            "FileIONetCDF object was open with 'FileIOBase::OpenMode::Write'.");
      }
      this->global_attributes.add_attribute(att_name, value);
    }

    //! get a NetCDfGlobalAtt from the FileIONetCDF object by its name
    const NetCDFGlobalAtt &
    read_global_attribute(const std::string & att_name) const;

    //! get a const std::vector<std::string> with the names of all current
    //! global attributes
    const std::vector<std::string> read_global_attribute_names() const;

    //! update the value/s or name of an exisiting (already written to the
    //! NetCDF file) global attribute. This is only allowed if the changes do
    //! not lead to an increas of the size of the global attribute and the
    //! data_type of the attribute is not changed. This function can only be
    //! used in open_mode = FileIOBase::OpenMode::Write or
    //! FileIOBase::OpenMode::Append.
    template <class T>
    void update_global_attribute(const std::string & old_att_name,
                                 const std::string & new_att_name,
                                 T new_att_value) {
      // calling this function is only allowed in write or append mode
      if (this->open_mode != FileIOBase::OpenMode::Write and
          this->open_mode != FileIOBase::OpenMode::Append) {
        throw FileIOError(
            "It is only possible to update global attributes when the "
            "FileIONetCDF object was open in 'FileIOBase::OpenMode::Write' or "
            "'FileIOBase::OpenMode::Append'.");
      }

      // for safety reasons the file should be in data mode because there the
      // header space cannot be expanded which might be time consuming.
      if (this->netcdf_mode != FileIONetCDF::NetCDFMode::DataMode) {
        // switch automatic into data mode and do not turn back into old mode
        int status_enddef{ncmu_enddef(this->netcdf_id)};
        if (status_enddef != NC_NOERR and status_enddef != NC_ENOTINDEFINE) {
          throw FileIOError(ncmu_strerror(status_enddef));
        }
        this->netcdf_mode = NetCDFMode::DataMode;
      }

      // find the old NetCDFGlobalAtt
      std::shared_ptr<NetCDFGlobalAtt> old_netcdf_global_attribute{
          this->global_attributes.set_global_attribute(old_att_name)};

      // check if the old NetCDFGlobalAtt was already written
      if (not old_netcdf_global_attribute->is_already_written_to_file()) {
        if (this->open_mode == FileIOBase::OpenMode::Write) {
          throw FileIOError(
              "You can only update a global attribute if it was already "
              "written to the NetCDF file. It seems like the the global "
              "attribute '" +
              old_att_name +
              "' was not written to the NetCDF file up to now. In "
              "FileIOBase::OpenMode::Write the global attributes are written "
              "during the first call of 'FileIONetCDF::write()' or when you "
              "close the file with 'FileIONetCDF::close()'.");
        } else if (this->open_mode == FileIOBase::OpenMode::Append) {
          throw FileIOError(
              "You can only update a global attribute if it was already "
              "written to the NetCDF file. It seems like the the global "
              "attribute '" +
              old_att_name +
              "' was not written to the NetCDF file. In "
              "'FileIOBase::OpenMode::Append' this is an unexpected "
              "behaviour and you should inform the programmers.");
        } else {
          throw FileIOError("Unexpected behaviour in "
                            "'FileIONetCDF::update_global_attribute()' pleas "
                            "inform the programmers.");
        }
      }

      // update the old global attribute value by the new one
      old_netcdf_global_attribute->update_global_attribute(new_att_name,
                                                           new_att_value);

      // rename the global attribute if the name has changed
      if (old_att_name != new_att_name) {
        int status{ncmu_rename_att(this->netcdf_id, NC_GLOBAL,
                                   old_att_name.c_str(), new_att_name.c_str())};
        if (status != NC_NOERR) {
          throw FileIOError(ncmu_strerror(status));
        }
      }
      // write the global attribute again to the NetCDF file
      int status{ncmu_put_att(this->netcdf_id, NC_GLOBAL,
                              old_netcdf_global_attribute->get_name().c_str(),
                              old_netcdf_global_attribute->get_data_type(),
                              old_netcdf_global_attribute->get_nelems(),
                              old_netcdf_global_attribute->get_value())};
      if (status != NC_NOERR) {
        throw FileIOError(ncmu_strerror(status));
      }
      this->netcdf_file_changes();
    }

   protected:
    //! open file for read/write
    //! This function is called by the constructor at instantiation.
    void open() final;

    //! register global field collection
    void register_field_collection_global(
        muGrid::GlobalFieldCollection & fc_global,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) final;

    //! register local field collection
    void register_field_collection_local(
        muGrid::LocalFieldCollection & fc_local,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) final;

    //! when registering the first global field collection, I initialise the
    //! global field collection local_pixels.
    void initialise_gfc_local_pixels(
        const muGrid::GlobalFieldCollection & fc_global);

    //! add the global pixels field associated with a local field collection to
    //! the global field collection which stores the position of the local
    //! pixels and the offsets to read the data of a local variable in a proper
    //! way.
    std::string
    register_lfc_to_gfc_local_pixels(muGrid::LocalFieldCollection & fc_local);

    //! write contents of all fields within the field collection with the name
    //! in field_names that have no frame dimension.
    void write_no_frame(const std::vector<std::string> & field_names);

    //! actual call of NetCDF functions to read in the data of a single
    //! NetCDFVarBase
    // void read_netcdfvar(const Index_t & frame, const NetCDFVarBase & var);

    //! define the dimensions in the NetCDF file (to write a file)
    void define_netcdf_dimensions(NetCDFDimensions & dimensions);

    //! define the variables in the NetCDF file (to write a file)
    void define_netcdf_variables(NetCDFVariables & variables);

    //! define the variables attributes in the NetCDF file (to write a file)
    void define_netcdf_attributes(NetCDFVariables & variables);

    //! inquiry and register the dimension ids of the NetCDF file (to read or
    //! append a file)
    //! @ ndims      : number of dimensions that have to be registered, i.e.
    //!                computed by ncmu_inq().
    //! @ unlimdimid : the NetCDF dimension ID of the unlimited dimension, i.e.
    //!                computed by ncmu_inq().
    void register_netcdf_dimension_ids(std::uint64_t ndims,
                                       Index_t unlimdimid);

    //! inquiry and register the variable ids of the NetCDF file (to read or
    //! append a file)
    //! @ ndims : number of variables that have to be registered, i.e. computed
    //!           by ncmu_inq().
    void register_netcdf_variable_ids(std::uint64_t nvars);

    //! inquiry and register the attribute names of variables from the NetCDF
    //! file. Here the names are registered because attributes have no unique
    //! IDs, their numbers can change. (to read or append a file)
    void register_netcdf_attribute_names();

    //! register the values of all attributes with registered names
    void register_netcdf_attribute_values();

    //! write NetCDFGlobalAtts from global_attributes (which are not already
    //! written) this function is only called if the file was open in
    //! FileIOBase::OpenMode::Write. Then it is only called twice, once in
    //! FileIONetCDF::open() and once in FileIONetCDF::write() or
    //! FileIONetCDF::close() if write() was not called before close.
    void define_global_attributes();

    //! call define global attributes and check all requiremnts and bring the
    //! NetCDF file in the correct status if necessary. If no save call is
    //! possible the function exits without doing anything
    void define_global_attributes_save_call();

    //! inquiry and register the global attribute names of variables from the
    //! NetCDF file. Here the names are registered because global attributes
    //! have no unique IDs, their numbers can change. (to read or append a file)
    void register_netcdf_global_attribute_names();

    //! register the values of all global attributes with registered names
    void register_netcdf_global_attribute_values();

    //! update the last modified flag by the current date and time
    void update_global_attribute_last_modified();

    //! save call of update_global_attribute_last_modified which checks all
    //! necessary conditions and otherwise do not call the function.
    void update_global_attribute_last_modified_save_call();

    //! stes the flag netcdf_file_changed to true if it is not already true
    void netcdf_file_changes();


    int netcdf_id{-1};  // the netcdf ID, -1 is an invalid value
    NetCDFMode netcdf_mode{
        NetCDFMode::UndefinedMode};  // save the modus of the NetCDF file e.g.
                                     // after ncmu_create it is in define_mode,
                                     // thus netcdf_mode = DefineMode.
    bool netcdf_file_changed{false};  // is set to true if FileIONetCDF changes
                                      // something in the NetCDF file
    bool global_attributes_defined{
        false};  // satisfy that global attributes are only written once.
                 // "false" means they are not written up to now.

    NetCDFGlobalAttributes global_attributes;
    NetCDFDimensions dimensions;
    NetCDFVariables variables;

    // for book keeping of local field collections
    const std::string pixel{"pixel"};
    const muGrid::FieldCollection::SubPtMap_t nb_sub_pts;
    muGrid::GlobalFieldCollection GFC_local_pixels;
    bool initialised_GFC_local_pixels{false};
    std::vector<std::string> written_local_pixel_fields{};
    std::vector<std::string> read_local_pixel_fields{};

    // for book keeping of state field fields
    std::vector<std::string>
        state_field_field_names{};  // stores field names of fields which are
                                    // already registered in state fields
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FILE_IO_NETCDF_HH_
