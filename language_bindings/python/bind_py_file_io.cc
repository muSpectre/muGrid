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

#include "libmugrid/file_io_base.hh"
#ifdef WITH_NETCDF_IO
#include "libmugrid/file_io_netcdf.hh"
#endif
#include "libmugrid/communicator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/eigen.h>

#include <string>
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
      .def("__getitem__", &FileIOBase::operator[], "frame_index"_a)
      .def("__len__", &FileIOBase::size)
      .def("__iter__",
           [](FileIOBase & file_io_base) {
             return py::make_iterator(file_io_base.begin(), file_io_base.end());
           })
      .def("append_frame", &FileIOBase::append_frame,
           py::return_value_policy::reference_internal)
      .def("get_communicator", &FileIOBase::get_communicator);
}

void add_file_frame(py::module & mod) {
  py::class_<FileFrame> file_frame(mod, "FileFrame");
  file_frame.def(py::init<FileIOBase &, Index_t>(), "parent"_a, "frame"_a)
      .def(
          "read",
          [](FileFrame & frame, const std::vector<std::string> & field_names) {
            return frame.read(field_names);
          },
          "field_names"_a)
      .def("read", [](FileFrame & frame) { return frame.read(); })
      .def(
          "write",
          [](FileFrame & frame, const std::vector<std::string> & field_names) {
            return frame.write(field_names);
          },
          "field_names"_a)
      .def("write", [](FileFrame & frame) { return frame.write(); });
}

#ifdef WITH_NETCDF_IO
void add_file_io_netcdf(py::module & mod) {
  py::class_<FileIONetCDF, FileIOBase> file_io(mod, "FileIONetCDF");
  file_io
      .def(py::init<const std::string &, const FileIOBase::OpenMode &,
                    Communicator>(),
           "file_name"_a, "open_mode"_a = FileIOBase::OpenMode::Read,
           "communicator"_a = Communicator())
      .def("close", &FileIONetCDF::close)
      .def("register_field_collection",
           &FileIONetCDF::register_field_collection, "field_collection"_a,
           "field_names"_a =
               std::vector<std::string>{muGrid::REGISTER_ALL_FIELDS},
           "state_field_unique_prefixes"_a =
               std::vector<std::string>{muGrid::REGISTER_ALL_STATE_FIELDS})
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
      .def(
          "write",
          [](FileIONetCDF & file_io_object, const Index_t & frame,
             const std::vector<std::string> & field_names) {
            file_io_object.write(frame, field_names);
          },
          "frame"_a, "field_names"_a)
      .def(
          "write",
          [](FileIONetCDF & file_io_object, const Index_t & frame) {
            file_io_object.write(frame);
          },
          "frame"_a)
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
            switch (file_io_object.read_global_attribute(g_att_name)
                        .get_data_type()) {
            case muGrid::MU_NC_CHAR: {
              const std::vector<char> & char_vec{
                  file_io_object.read_global_attribute(g_att_name)
                      .get_typed_value_c()};
              return py::cast(std::string(char_vec.begin(), char_vec.end()));
              break;
            }
            case muGrid::MU_NC_INT: {
              return py::cast(file_io_object.read_global_attribute(g_att_name)
                                  .get_typed_value_i());
              break;
            }
            case muGrid::MU_NC_UINT: {
              return py::cast(file_io_object.read_global_attribute(g_att_name)
                                  .get_typed_value_ui());
              break;
            }
            case muGrid::MU_NC_INDEX_T: {
              return py::cast(file_io_object.read_global_attribute(g_att_name)
                                  .get_typed_value_l());
              break;
            }
            case muGrid::MU_NC_REAL: {
              return py::cast(file_io_object.read_global_attribute(g_att_name)
                                  .get_typed_value_d());
              break;
            }
            default:
              throw muGrid::FileIOError(
                  "Unknown data type of global attribute '" + g_att_name +
                  "'value in the FileIONetCDF python binding "
                  "'read_global_attribute()'.");
            }
          },
          "att_name"_a)
      .def("read_global_attributes",
           [](FileIONetCDF & file_io_object) {
             auto d = py::dict();
             for (auto & g_att_name :
                  file_io_object.read_global_attribute_names()) {
               // switch to the correct function of the family
               // ".get_typed_value_*()"
               switch (file_io_object.read_global_attribute(g_att_name)
                           .get_data_type()) {
               case muGrid::MU_NC_CHAR: {
                 const std::vector<char> & char_vec{
                     file_io_object.read_global_attribute(g_att_name)
                         .get_typed_value_c()};
                 d[py::str(g_att_name)] =
                     std::string(char_vec.begin(), char_vec.end());
                 break;
               }
               case muGrid::MU_NC_INT: {
                 d[py::str(g_att_name)] =
                     file_io_object.read_global_attribute(g_att_name)
                         .get_typed_value_i();
                 break;
               }
               case muGrid::MU_NC_UINT: {
                 d[py::str(g_att_name)] =
                     file_io_object.read_global_attribute(g_att_name)
                         .get_typed_value_ui();
                 break;
               }
               case muGrid::MU_NC_INDEX_T: {
                 d[py::str(g_att_name)] =
                     file_io_object.read_global_attribute(g_att_name)
                         .get_typed_value_l();
                 break;
               }
               case muGrid::MU_NC_REAL: {
                 d[py::str(g_att_name)] =
                     file_io_object.read_global_attribute(g_att_name)
                         .get_typed_value_d();
                 break;
               }
               default:
                 throw muGrid::FileIOError(
                     "Unknown data type of global attribute '" + g_att_name +
                     "'value in the FileIONetCDF python binding "
                     "'read_global_attributes()'.");
               }
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
