/**
 * @file   numpy_tools.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Dec 2019
 *
 * @brief  Convenience functionality for working with (py::'s) numpy arrays.
 *         These are implemented header-only, in order to avoid an explicit
 *         dependency on py::
 *
 * Copyright © 2018 Lars Pastewka, Till Junge
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

#ifndef SRC_LIBMUGRID_NUMPY_TOOLS_HH_
#define SRC_LIBMUGRID_NUMPY_TOOLS_HH_

#include "field_typed.hh"
#include "raw_memory_operations.hh"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

namespace muGrid {
    /**
     * base class for numpy related exceptions
     */
    class NumpyError : public RuntimeError {
    public:
        //! constructor
        explicit NumpyError(const std::string &what) : RuntimeError(what) {
        }

        //! constructor
        explicit NumpyError(const char *what) : RuntimeError(what) {
        }
    };

    /* Wrap a field into a numpy array, without copying the data */
    template<typename T>
    py::array_t<T, py::array::f_style>
    numpy_wrap(const TypedFieldBase<T> &field,
               IterUnit iter_type = IterUnit::SubPt,
               Shape_t shape = {},
               Shape_t offset = {}) {
        if (shape.empty()) {
            shape = field.get_shape(iter_type);
        }
        Shape_t strides1{field.get_strides(iter_type, 1)};
        if (offset.empty()) {
            offset.resize(strides1.size());
        }
        Index_t buffer_offset{0};
        for (auto &&tup: akantu::zip(strides1, offset)) {
            auto &&s{std::get<0>(tup)};
            auto &&o{std::get<1>(tup)};
            buffer_offset += o * s;
        }
        Shape_t strides{field.get_strides(iter_type, sizeof(T))};
        return py::array_t<T, py::array::f_style>(shape, strides, field.data() + buffer_offset,
                                                  py::capsule([]() {}));
    }

    /* Copy a field into a numpy array */
    template<typename T>
    py::array_t<T, py::array::f_style>
    numpy_copy(const TypedFieldBase<T> &field,
               IterUnit iter_type = IterUnit::SubPt) {
        const Shape_t shape{field.get_shape(iter_type)};
        py::array_t<T> array(shape);
        Shape_t array_strides(array.strides(), array.strides() + array.ndim());
        // numpy arrays have stride in bytes
        for (auto &&s: array_strides)
            s /= sizeof(T);
        muGrid::raw_mem_ops::strided_copy(shape, field.get_strides(iter_type),
                                          array_strides, field.data(),
                                          array.mutable_data());
        return std::move(array);
    }

    /* Turn any type that can be enumerated into a tuple */
    template<typename T>
    py::tuple to_tuple(T a) {
        py::tuple t(a.get_dim());
        ssize_t i = 0;
        for (auto &&v: a) {
            t[i] = v;
            i++;
        }
        return t;
    }
} // namespace muGrid

#endif  // SRC_LIBMUGRID_NUMPY_TOOLS_HH_
