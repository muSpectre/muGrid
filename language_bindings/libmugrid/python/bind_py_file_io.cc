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
  file_io_base
      .def(py::init<const std::string &, const FileIOBase::OpenMode &,
                    Communicator>(),
           "file_name"_a, "open_mode"_a,
#ifdef WITH_MPI
           "communicator"_a)
#else
           "communicator"_a = Communicator())
#endif
      .def("__getitem__", &FileIOBase::operator[], "frame_index"_a)
      .def("__len__", &FileIOBase::size)
      .def("__iter__",
           [](FileIOBase & file_io_base) {
             return py::make_iterator(file_io_base.begin(), file_io_base.end());
           })
      .def("append_frame", &FileIOBase::append_frame,
           py::return_value_policy::reference_internal)
      .def("get_communicator", &FileIOBase::get_communicator);

  py::enum_<FileIOBase::OpenMode>(file_io_base, "OpenMode")
      .value("Read", FileIOBase::OpenMode::Read)
      .value("Write", FileIOBase::OpenMode::Write)
      .value("Append", FileIOBase::OpenMode::Append)
      .export_values();
}

void add_file_frame(py::module & mod) {
  py::class_<FileFrame> file_frame(mod, "FileFrame");
  file_frame.def(py::init<FileIOBase &, Index_t>(), "parent"_a, "frame"_a)
      .def("read",
           [](FileFrame & frame, const std::vector<std::string> & field_names) {
             return frame.read(field_names);
           },
           "field_names"_a)
      .def("read", [](FileFrame & frame) { return frame.read(); })
      .def("write",
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
           "file_name"_a, "open_mode"_a,
#ifdef WITH_MPI
           "communicator"_a)
#else
           "communicator"_a = Communicator())
#endif
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
          "frame"_a);
}
#endif

void add_file_io_classes(py::module & mod) {
  add_file_io_base(mod);
  add_file_frame(mod);
#ifdef WITH_NETCDF_IO
  add_file_io_netcdf(mod);
#endif
}
