/**
 * @file   bind_py_options_dictionary.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   30 Nov 2021
 *
 * @brief  python bindings for the python dictionary-like io class Dictionary
 *
 * Copyright © 2021 Till Junge
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
#include "libmugrid/options_dictionary.hh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <string>
#include <memory>
#include <functional>

namespace py = pybind11;
using pybind11::literals::operator""_a;
using muGrid::Dictionary;
using muGrid::DictionaryError;
using muGrid::RuntimeValue;
using muGrid::Int;
using muGrid::Real;

Dictionary convert(const py::dict & d) {
  Dictionary ret_dict{};
  for (auto && item : d) {
    if (not py::isinstance<py::str>(item.first)) {
      throw muGrid::DictionaryError("keys must be strings");
    }
    auto && key{std::string(py::str(item.first))};
    auto && value{item.second};
    if (py::isinstance<py::int_>(value)) {
      ret_dict.add(key, value.cast<Int>());
    } else if (py::isinstance<py::float_>(value)) {
      ret_dict.add(key, value.cast<Real>());
    } else if (py::isinstance<py::dict>(value)) {
      auto val{value.cast<py::dict>()};
      ret_dict.add(key, convert(val));
    } else if (py::isinstance<py::array_t<Real>>(value)) {
      ret_dict.add(key, value.cast<py::EigenDRef<Eigen::MatrixXd>>());
    } else {
      throw DictionaryError("Unknown python type used in dict");
    }
  }
  return ret_dict;
}

void add_options_dictionary(py::module & mod) {
  using Ref_t =
      py::EigenDRef<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>;

  py::class_<Dictionary>(mod, "Dictionary")
      .def(py::init<>())
      .def(py::init<const std::string &, const Int &>(), "key"_a, "value"_a)
      .def(py::init<const std::string &, const Real &>(), "key"_a, "value"_a)
      .def(py::init([](const std::string & key,
                       const Ref_t val) -> std::unique_ptr<Dictionary> {
             return std::make_unique<Dictionary>(key, val);
           }),
           "key"_a, "value"_a)
      .def(py::init([](py::dict & dict) -> std::unique_ptr<Dictionary> {
             Dictionary ret_dict{convert(dict)};
             return std::make_unique<Dictionary>(ret_dict);
           }),
           "dict"_a)
      .def(
          "__getitem__",
          [](const Dictionary & dict, const std::string & key) -> py::object {
            auto && retval{dict[key]};
            switch (retval.get_value_type()) {
            case RuntimeValue::ValueType::Int: {
              return py::int_(retval.get_int());
              break;
            }
            case RuntimeValue::ValueType::Real: {
              return py::float_(retval.get_real());
              break;
            }
            case RuntimeValue::ValueType::Matrix: {
              auto && matrix{retval.get_matrix()};
              std::array<muGrid::Index_t, 2> shape{matrix.rows(),
                                                   matrix.cols()};
              py::array_t<Real> retval(shape);
              for (int i{0}; i < matrix.rows(); ++i) {
                for (int j{0}; j < matrix.cols(); ++j) {
                  retval.mutable_at(i, j) = matrix(i, j);
                }
              }
              return std::move(retval);
              break;
            }
            case RuntimeValue::ValueType::Dictionary: {
              throw DictionaryError(
                  "nested dictionaries are not yet supported");
              break;
            }
            default:
              throw DictionaryError("Unknown variable type");
              break;
            }
          },
          "key"_a)
      .def(
          "add",
          [](Dictionary & d, const std::string & key,
             const Int & value) -> void { d.add(key, value); },
          "key"_a, "value"_a)
      .def(
          "add",
          [](Dictionary & d, const std::string & key,
             const Real & value) -> void { d.add(key, value); },
          "key"_a, "value"_a)
      .def(
          "add",
          [](Dictionary & d, const std::string & key,
             const py::EigenDRef<Eigen::MatrixXd> & value) -> void {
            d.add(key, value);
          },
          "key"_a, "value"_a)
      .def(
          "__setitem__",
          [](Dictionary & dict, const std::string & key,
             py::object & value) {
            if (py::isinstance<py::int_>(value)) {
              dict[key] = value.cast<Int>();
            } else if (py::isinstance<py::float_>(value)) {
              dict[key] = value.cast<Real>();
            } else if (py::isinstance<py::dict>(value)) {
              auto val{value.cast<py::dict>()};
              dict[key] = convert(val);
            } else if (py::isinstance<py::array_t<Real>>(value)) {
              dict[key] = value.cast<py::EigenDRef<Eigen::MatrixXd>>();
            } else {
              throw DictionaryError("Unknown python type used in dict");
            }
          },
          "key"_a, "value"_a);
}
