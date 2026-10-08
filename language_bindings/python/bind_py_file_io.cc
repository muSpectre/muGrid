/**
 * @file   bind_py_file_io.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   11 Sep 2020
 *
 * @brief  Python bindings for the file I/O
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "io/file_io_base.hh"
#ifdef WITH_NETCDF_IO
#include "io/file_io_netcdf.hh"
#endif
#include "mpi/communicator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using muGrid::FileIOBase;
using muGrid::FileFrame;
using muGrid::Communicator;
using muGrid::Index_t;
using pybind11::literals::operator""_a;

#ifdef WITH_NETCDF_IO
using muGrid::FileIONetCDF;
#endif

namespace py = pybind11;

#ifdef WITH_NETCDF_IO
namespace {
  /**
   * Convert a NetCDF global attribute to the matching Python object, selected
   * by its stored data type. Shared by the read_global_attribute() and
   * read_global_attributes() bindings, which otherwise duplicated this switch.
   */
  py::object attribute_to_py(const muGrid::NetCDFGlobalAtt & att,
                             const std::string & att_name) {
    switch (att.get_data_type()) {
    case muGrid::MU_NC_CHAR: {
      const std::vector<char> & char_vec{att.get_typed_value_c()};
      return py::cast(std::string(char_vec.begin(), char_vec.end()));
    }
    case muGrid::MU_NC_INT:
      return py::cast(att.get_typed_value_i());
    case muGrid::MU_NC_UINT:
      return py::cast(att.get_typed_value_ui());
    case muGrid::MU_NC_INDEX_T:
      return py::cast(att.get_typed_value_l());
    case muGrid::MU_NC_REAL:
      return py::cast(att.get_typed_value_d());
    default:
      throw muGrid::FileIOError(
          "Unknown data type of global attribute '" + att_name +
          "' value in the FileIONetCDF python bindings.");
    }
  }

  /**
   * Map a numpy dtype to the matching NetCDF external data type, for
   * per-frame variables.
   */
  nc_type numpy_dtype_to_nc_type(const py::dtype & dt) {
    const char kind{dt.kind()};
    const py::ssize_t size{dt.itemsize()};
    if (kind == 'f' && size == 8) return NC_DOUBLE;
    if (kind == 'f' && size == 4) return NC_FLOAT;
    if (kind == 'i' && size == 8) return NC_INT64;
    if (kind == 'i' && size == 4) return NC_INT;
    if (kind == 'i' && size == 2) return NC_SHORT;
    if (kind == 'u' && size == 8) return NC_UINT64;
    if (kind == 'u' && size == 4) return NC_UINT;
    if (kind == 'u' && size == 2) return NC_USHORT;
    throw muGrid::FileIOError(
        "Unsupported numpy dtype for a per-frame variable (kind '" +
        std::string(1, kind) + "', itemsize " + std::to_string(size) + ").");
  }

  /**
   * Inverse of numpy_dtype_to_nc_type, for the types it produces.
   */
  py::dtype nc_type_to_numpy_dtype(const nc_type & type) {
    switch (type) {
    case NC_DOUBLE:
      return py::dtype::of<double>();
    case NC_FLOAT:
      return py::dtype::of<float>();
    case NC_INT64:
      return py::dtype::of<std::int64_t>();
    case NC_INT:
      return py::dtype::of<std::int32_t>();
    case NC_SHORT:
      return py::dtype::of<std::int16_t>();
    case NC_UINT64:
      return py::dtype::of<std::uint64_t>();
    case NC_UINT:
      return py::dtype::of<std::uint32_t>();
    case NC_USHORT:
      return py::dtype::of<std::uint16_t>();
    default:
      throw muGrid::FileIOError("Unsupported NetCDF data type " +
                                std::to_string(type) +
                                " for a per-frame variable.");
    }
  }

  /**
   * A numpy array that views the host buffer of the per-frame variable
   * `name`. The array's base keeps the FileIONetCDF alive, so the view stays
   * valid as long as it is used.
   */
  py::array frame_variable_view(FileIONetCDF & file_io_object,
                                const std::string & name) {
    muGrid::NetCDFVarFrameData & var{file_io_object.get_frame_variable(name)};
    py::dtype dt{nc_type_to_numpy_dtype(var.get_data_type())};
    const auto & shape{var.get_component_shape()};
    std::vector<py::ssize_t> np_shape(shape.begin(), shape.end());
    std::vector<py::ssize_t> strides(np_shape.size());
    py::ssize_t stride{dt.itemsize()};
    for (size_t i{np_shape.size()}; i-- > 0;) {
      strides[i] = stride;
      stride *= np_shape[i];
    }
    return py::array(dt, np_shape, strides, var.get_buf(),
                     py::cast(&file_io_object));
  }

  /**
   * Store keyword values in the buffers of the per-frame variables they name
   * and return those names. Every value is checked (name, shape, and a
   * same-kind cast to the variable's dtype) before any buffer is touched, so
   * a rejected call leaves all buffers unchanged.
   */
  std::vector<std::string> set_frame_values(FileIONetCDF & file_io_object,
                                            const py::kwargs & values) {
    py::module_ np{py::module_::import("numpy")};
    std::vector<std::string> names{};
    std::vector<std::pair<py::array, py::object>> assignments{};
    for (const auto & item : values) {
      const std::string name{py::cast<std::string>(item.first)};
      py::array view{frame_variable_view(file_io_object, name)};
      py::object value{np.attr("asarray")(item.second)};
      if (!py::cast<bool>(np.attr("can_cast")(value.attr("dtype"),
                                              view.attr("dtype"),
                                              "same_kind"))) {
        throw py::type_error(
            "Cannot store a value of dtype " +
            py::cast<std::string>(py::str(value.attr("dtype"))) +
            " in the per-frame variable '" + name + "' of dtype " +
            py::cast<std::string>(py::str(view.attr("dtype"))) + ".");
      }
      // raises ValueError if the value does not fit the variable's shape
      np.attr("broadcast_to")(value, view.attr("shape"));
      names.push_back(name);
      assignments.emplace_back(view, value);
    }
    for (auto & [view, value] : assignments) {
      np.attr("copyto")(view, value, "casting"_a = "same_kind");
    }
    return names;
  }

  /**
   * Write frame `frame`: store the keyword values (see set_frame_values),
   * then write `field_names` plus the variables given as keywords, or every
   * registered variable if `field_names` is None.
   */
  void write_frame(FileIONetCDF & file_io_object, const Index_t & frame,
                   const py::object & field_names, const py::kwargs & values) {
    const std::vector<std::string> value_names{
        set_frame_values(file_io_object, values)};
    if (field_names.is_none()) {
      file_io_object.write(frame);
      return;
    }
    auto names{py::cast<std::vector<std::string>>(field_names)};
    for (const auto & name : value_names) {
      if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
      }
    }
    file_io_object.write(frame, names);
  }

  constexpr const char * write_frame_doc{
      R"(Write a frame. Per-frame variables can be given their value for this
frame as keyword arguments, e.g. write(iteration=12, objective=0.5); a value is
stored in the variable's buffer, so it stays there for later frames until it
is set again. All keyword values are checked (known per-frame variable, shape,
same-kind cast to its dtype, so a float is not silently truncated into an
integer variable) before anything is stored. With `field_names`, only those
fields/variables are written, plus the ones given as keywords; without it,
every registered field and variable is written.)"};
}  // namespace
#endif  // WITH_NETCDF_IO

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
class PyFileIOBase : public FileIOBase {
 public:
  /* Inherit the constructors */
  using FileIOBase::FileIOBase;

  // Trampoline for virtual functions
  void register_field_collection(
      muGrid::FieldCollection & fc, std::vector<std::string> field_names,
      std::vector<std::string> state_field_unique_prefixes) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, register_field_collection, fc,
                           field_names, state_field_unique_prefixes);
  }

  void close() override { PYBIND11_OVERLOAD_PURE(void, FileIOBase, close); }

  void read(const Index_t & frame,
            const std::vector<std::string> & field_names) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, read, frame, field_names);
  }

  void read(const Index_t & frame) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, read, frame);
  }

  void write(const Index_t & frame,
             const std::vector<std::string> & field_names) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, write, frame, field_names);
  }

  void write(const Index_t & frame) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, write, frame);
  }

  void open() override { PYBIND11_OVERLOAD_PURE(void, FileIOBase, open); };

  void register_field_collection_global(
      muGrid::GlobalFieldCollection & fc_global,
      const std::vector<std::string> & field_names,
      const std::vector<std::string> & state_field_unique_prefixes) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, register_field_collection_global,
                           fc_global, field_names, state_field_unique_prefixes);
  }

  void register_field_collection_local(
      muGrid::LocalFieldCollection & fc_local,
      const std::vector<std::string> & field_names,
      const std::vector<std::string> & state_field_unique_prefixes) override {
    PYBIND11_OVERLOAD_PURE(void, FileIOBase, register_field_collection_local,
                           fc_local, field_names, state_field_unique_prefixes);
  }
};

void add_file_io_base(py::module & mod) {
  py::class_<FileIOBase, PyFileIOBase> file_io_base(mod, "FileIOBase");

  py::enum_<FileIOBase::OpenMode>(file_io_base, "OpenMode")
      .value("Read", FileIOBase::OpenMode::Read)
      .value("Write", FileIOBase::OpenMode::Write)
      .value("Overwrite", FileIOBase::OpenMode::Overwrite)
      .value("Append", FileIOBase::OpenMode::Append)
      .export_values();

  file_io_base
      .def(py::init<const std::string &, const FileIOBase::OpenMode &,
                    Communicator>(),
           "file_name"_a, "open_mode"_a = FileIOBase::OpenMode::Read,
           "communicator"_a = Communicator())
      // FileFrame objects hold a reference to their parent FileIOBase, so
      // everything handing one out pins the file object (keep_alive<0, 1>).
      .def("__getitem__", &FileIOBase::operator[], "frame_index"_a,
           py::keep_alive<0, 1>())
      .def("__len__", &FileIOBase::size)
      .def(
          "__iter__",
          [](FileIOBase & file_io_base) {
            return py::make_iterator(file_io_base.begin(), file_io_base.end());
          },
          py::keep_alive<0, 1>())
      .def("append_frame", &FileIOBase::append_frame,
           py::return_value_policy::reference_internal)
      .def_property_readonly("communicator", &FileIOBase::get_communicator);
}

void add_file_frame(py::module & mod) {
  py::class_<FileFrame> file_frame(mod, "FileFrame");
  // The frame references its parent FileIOBase; keep the parent alive for
  // the frame's lifetime.
  file_frame
      .def(py::init<FileIOBase &, Index_t>(), "parent"_a, "frame"_a,
           py::keep_alive<1, 2>())
      .def(
          "read",
          [](FileFrame & frame, const std::vector<std::string> & field_names) {
            return frame.read(field_names);
          },
          "field_names"_a)
      .def("read", [](FileFrame & frame) { return frame.read(); })
      .def(
          "write",
          [](FileFrame & frame, const py::object & field_names,
             const py::kwargs & values) {
#ifdef WITH_NETCDF_IO
            auto * netcdf{dynamic_cast<FileIONetCDF *>(&frame.get_parent())};
            if (netcdf != nullptr) {
              write_frame(*netcdf, frame.get_frame(), field_names, values);
              return;
            }
#endif  // WITH_NETCDF_IO
            if (values.size() > 0) {
              throw muGrid::FileIOError(
                  "Keyword values are only supported for per-frame "
                  "variables of a FileIONetCDF.");
            }
            if (field_names.is_none()) {
              frame.write();
            } else {
              frame.write(py::cast<std::vector<std::string>>(field_names));
            }
          },
          "field_names"_a = py::none(),
#ifdef WITH_NETCDF_IO
          write_frame_doc
#else
          "Write a frame: `field_names`, or every registered field if None."
#endif  // WITH_NETCDF_IO
      );
}

#ifdef WITH_NETCDF_IO
void add_file_io_netcdf(py::module & mod) {
  // The NetCDF library backend and its version string, for diagnostics. PnetCDF
  // is used with MPI, serial Unidata NetCDF otherwise.
#ifdef WITH_MPI
  mod.attr("netcdf_backend") = std::string("PnetCDF");
  mod.attr("netcdf_version") = std::string(ncmpi_inq_libvers());
#else
  mod.attr("netcdf_backend") = std::string("NetCDF");
  mod.attr("netcdf_version") = std::string(nc_inq_libvers());
#endif

  py::class_<FileIONetCDF, FileIOBase> file_io(mod, "FileIONetCDF");
  file_io
      .def(py::init<const std::string &, const FileIOBase::OpenMode &,
                    Communicator>(),
           "file_name"_a, "open_mode"_a = FileIOBase::OpenMode::Read,
           "communicator"_a = Communicator())
      .def("sync", &FileIONetCDF::sync,
           "Flush buffered data to disk so a concurrent reader sees the "
           "frames written so far. Call after append_frame() to checkpoint "
           "incrementally; the PnetCDF backend already commits collective "
           "writes, the classic/NetCDF-4 backend may otherwise buffer until "
           "close().")
      .def("close", &FileIONetCDF::close)
      .def("register_field_collection",
           &FileIONetCDF::register_field_collection, "field_collection"_a,
           "field_names"_a =
               std::vector<std::string>{muGrid::REGISTER_ALL_FIELDS},
           "state_field_unique_prefixes"_a =
               std::vector<std::string>{muGrid::REGISTER_ALL_STATE_FIELDS})
      .def(
          "register_frame_variable",
          [](FileIONetCDF & file_io_object, const std::string & name,
             std::vector<IOSize_t> shape, py::object dtype) {
            // Register the (grid-less) per-frame variable and return a numpy
            // array that *views* its host buffer, so the caller sets the value
            // for the current frame by writing into it and then calling
            // write()/append_frame().
            py::dtype dt{py::dtype::from_args(dtype)};
            file_io_object.register_frame_variable(
                name, shape, numpy_dtype_to_nc_type(dt));
            return frame_variable_view(file_io_object, name);
          },
          "name"_a, "shape"_a, "dtype"_a,
          R"(Register a per-frame, grid-less quantity (a small tensor with one
value for the whole domain per frame, replicated across MPI ranks) and return a
numpy array that views its buffer. Call before the first frame is written; set
the current frame's value by passing it to write() as a keyword argument, e.g.
append_frame().write(name=value), or by writing into the returned array before
write(). Args: name (str), shape (sequence of int, the shape of a single
frame's value; [] for a scalar), dtype (numpy dtype). See also
write_global_attribute for values that do not vary per frame.)")
      .def("frame_variable", &frame_variable_view, "name"_a,
           R"(Return a numpy array that views the buffer of the registered
per-frame variable `name`: the value the next write() stores, or the one the
last read() loaded.)")
      .def(
          "read",
          [](FileIONetCDF & file_io_object, const Index_t & frame,
             const std::vector<std::string> & field_names) {
            file_io_object.read(frame, field_names);
          },
          "frame"_a, "field_names"_a)
      .def(
          "read",
          [](FileIONetCDF & file_io_object, const Index_t & frame) {
            file_io_object.read(frame);
          },
          "frame"_a)
      .def("write", &write_frame, "frame"_a, "field_names"_a = py::none(),
           write_frame_doc)
      .def("write_global_attribute",
           &FileIONetCDF::write_global_attribute<std::string &>, "att_name"_a,
           "value"_a)
      .def("write_global_attribute",
           &FileIONetCDF::write_global_attribute<std::vector<muGrid::Int> &>,
           "att_name"_a, "value"_a)
      .def("write_global_attribute",
           &FileIONetCDF::write_global_attribute<std::vector<muGrid::Uint> &>,
           "att_name"_a, "value"_a)
      .def(
          "write_global_attribute",
          &FileIONetCDF::write_global_attribute<std::vector<muGrid::Index_t> &>,
          "att_name"_a, "value"_a)
      .def("write_global_attribute",
           &FileIONetCDF::write_global_attribute<std::vector<muGrid::Real> &>,
           "att_name"_a, "value"_a)
      .def("read_global_attribute_names",
           &FileIONetCDF::read_global_attribute_names)
      .def(
          "read_global_attribute",
          [](FileIONetCDF & file_io_object, std::string & g_att_name) {
            return attribute_to_py(
                file_io_object.read_global_attribute(g_att_name), g_att_name);
          },
          "att_name"_a)
      .def("read_global_attributes",
           [](FileIONetCDF & file_io_object) {
             auto d = py::dict();
             for (auto & g_att_name :
                  file_io_object.read_global_attribute_names()) {
               d[py::str(g_att_name)] = attribute_to_py(
                   file_io_object.read_global_attribute(g_att_name),
                   g_att_name);
             }
             return d;
           })
      .def("update_global_attribute",
           &FileIONetCDF::update_global_attribute<std::string &>,
           "old_att_name"_a, "new_att_name"_a, "value"_a)
      .def("update_global_attribute",
           &FileIONetCDF::update_global_attribute<std::vector<muGrid::Int> &>,
           "old_att_name"_a, "new_att_name"_a, "value"_a)
      .def("update_global_attribute",
           &FileIONetCDF::update_global_attribute<std::vector<muGrid::Uint> &>,
           "old_att_name"_a, "new_att_name"_a, "value"_a)
      .def("update_global_attribute",
           &FileIONetCDF::update_global_attribute<
               std::vector<muGrid::Index_t> &>,
           "old_att_name"_a, "new_att_name"_a, "value"_a)
      .def("update_global_attribute",
           &FileIONetCDF::update_global_attribute<std::vector<muGrid::Real> &>,
           "old_att_name"_a, "new_att_name"_a, "value"_a);
}
#endif

void add_file_io_classes(py::module & mod) {
  add_file_io_base(mod);
  add_file_frame(mod);
#ifdef WITH_NETCDF_IO
  add_file_io_netcdf(mod);
#endif
}
