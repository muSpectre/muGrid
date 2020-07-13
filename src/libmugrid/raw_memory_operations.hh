/**
 * @file   raw_memory_operations.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 May 2020
 *
 * @brief  functions for unsafe raw memory operations. Use these only when
 *         necessary
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

#include "grid_common.hh"
#include "exception.hh"

#include <cstring>
#include <numeric>
#include <vector>

#ifndef SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_
#define SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_

namespace muGrid {
  namespace raw_mem_ops {
    /* ---------------------------------------------------------------------- */
    inline size_t prod(const Shape_t & vec) {
      return std::accumulate(vec.begin(), vec.end(), 1,
                             std::multiplies<Index_t>());
    }

    /* ---------------------------------------------------------------------- */
    inline size_t linear_index(const Shape_t & index,
                               const Shape_t & strides) {
      return std::inner_product(index.begin(), index.end(), strides.begin(),
                                size_t{});
    }

    class CartesianContainer {
     public:
      //! Default constructor
      CartesianContainer() = delete;

      //! Constructor from shape
      explicit CartesianContainer(const Shape_t & shape,
                                  const Shape_t & strides);

      //! Copy constructor
      CartesianContainer(const CartesianContainer & other) = delete;

      //! Move constructor
      CartesianContainer(CartesianContainer && other) = default;

      //! Destructor
      virtual ~CartesianContainer() = default;

      //! Copy assignment operator
      CartesianContainer & operator=(const CartesianContainer & other) = delete;

      //! Move assignment operator
      CartesianContainer & operator=(CartesianContainer && other) = default;

      Index_t get_nb_dim() const { return this->shape.size(); }

      class iterator {
       public:
        explicit iterator(const CartesianContainer & container,
                          const size_t & counter = 0)
            : container{container},
              index(this->container.shape.size()), counter{counter} {}

        ~iterator() {}

        iterator & operator++() {
          const auto & axes_order{this->container.axes_order};
          ++this->counter;
          this->index[axes_order[0]]++;
          for (size_t i{1}; i < this->index.size(); ++i) {
            this->index[axes_order[i]] +=
                index[axes_order[i - 1]] /
                this->container.shape[axes_order[i - 1]];
          }
          for (size_t i{0}; i < this->index.size(); ++i) {
            this->index[i] %= this->container.shape[i];
          }
          return *this;
        }

        const Shape_t & operator*() const { return this->index; }

        bool operator!=(const iterator & other) const {
          return this->counter != other.counter;
        }

       protected:
        const CartesianContainer & container;
        Shape_t index;
        size_t counter{0};
      };

      iterator begin() const { return iterator{*this}; }
      iterator end() const { return iterator{*this, prod(this->shape)}; }

     protected:
      Shape_t shape, axes_order;
    };

    /* ---------------------------------------------------------------------- */
    template <typename T>
    void strided_copy(const Shape_t & logical_shape,
                      const Shape_t & input_strides,
                      const Shape_t & output_strides,
                      const T * input_data, T * output_data) {
      if (logical_shape.size() != input_strides.size()) {
        std::stringstream message{};
        message << "Dimensions mismatch: The shape " << logical_shape
                << " is of dimension " << logical_shape.size()
                << " but the input_strides " << input_strides
                << " are of dimension " << input_strides.size() << ".";
        throw RuntimeError{message.str()};
      }
      if (logical_shape.size() != output_strides.size()) {
        std::stringstream message{};
        message << "Dimensions mismatch: The shape " << logical_shape
                << " is of dimension " << logical_shape.size()
                << " but the output_strides " << output_strides
                << " are of dimension " << output_strides.size() << ".";
        throw RuntimeError{message.str()};
      }
      CartesianContainer indices{logical_shape, output_strides};
      for (auto && index : indices) {
        auto && read_from{linear_index(index, input_strides)};
        auto && write_to{linear_index(index, output_strides)};
        output_data[write_to] = input_data[read_from];
      }
    }
  }  // namespace raw_mem_ops
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_
