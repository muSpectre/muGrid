/**
 * @file   nfield_map.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Aug 2019
 *
 * @brief  Implementation of the base class of all field maps
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
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
 */

#ifndef SRC_LIBMUGRID_NFIELD_MAP_HH_
#define SRC_LIBMUGRID_NFIELD_MAP_HH_

#include "grid_common.hh"

#include <type_traits>

namespace muGrid {
  /**
   * base class for field map-related exceptions
   */
  class NFieldMapError : public std::runtime_error {
   public:
    //! constructor
    explicit NFieldMapError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit NFieldMapError(const char * what) : std::runtime_error(what) {}
  };

  // forward declaration
  template <typename T>
  class TypedNFieldBase;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField>
  class NFieldMap {
   public:
    using NField_t = std::conditional_t<ConstField, const TypedNFieldBase<T>,
                                        TypedNFieldBase<T>>;
    template <bool ConstVal>
    using value_type = std::conditional_t<
        ConstVal,
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>,
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>;
    //! Default constructor
    NFieldMap() = delete;

    /**
     * Constructor from a field. The default case is a map iterating over
     * quadrature points with a matrix of shape (nb_components × 1) per field
     * entry
     */
    explicit NFieldMap(NField_t & field,
                       Iteration iter_type = Iteration::QuadPt);

    /**
     * Constructor from a field with explicitly chosen shape of iterate. (the
     * number of columns is inferred).
     */
    NFieldMap(NField_t & field, Dim_t nb_rows,
              Iteration iter_type = Iteration::QuadPt);

    //! Copy constructor
    NFieldMap(const NFieldMap & other) = default;

    //! Move constructor
    NFieldMap(NFieldMap && other) = default;

    //! Destructor
    virtual ~NFieldMap() = default;

    //! Copy assignment operator (delete because of reference member)
    NFieldMap & operator=(const NFieldMap & other) = delete;

    //! Move assignment operator (delete because of reference member)
    NFieldMap & operator=(NFieldMap && other) = delete;

    template <bool ConstIter>
    class Iterator;
    using iterator = Iterator<false or ConstField>;
    using const_iterator = Iterator<true>;

    iterator begin();
    iterator end();
    const_iterator cbegin();
    const_iterator cend();
    const_iterator begin() const;
    const_iterator end() const;

    /**
     * returns the number of iterates produced by this map (corresponds to the
     * number of field entries if Iteration::Quadpt, or the number of
     * pixels/voxels if Iteration::Pixel);
     */
    size_t size() const;

    value_type<ConstField> operator[](size_t index) {
      return value_type<ConstField>{this->data_ptr + index * this->stride,
                                    this->nb_rows, this->nb_cols};
    }
    value_type<true> operator[](size_t index) const {
      return value_type<true>{this->data_ptr + index * this->stride,
                              this->nb_rows, this->nb_cols};
    }

    //! query the size from the field's collection and set data_ptr
    void initialise();

   protected:
    //! mapped field. Needed for query at initialisations
    const NField_t & field;
    const Iteration iteration;  //!< type of map iteration
    const Dim_t stride;         //!< precomputed stride
    const Dim_t nb_rows;        //!< number of rows of the iterate
    const Dim_t nb_cols;        //!< number of columns fo the iterate
    /**
     * Pointer to mapped data; is also unknown at construction and set in the
     * map's begin function
     */
    T * data_ptr{nullptr};
    bool is_initialised{false};
  };

  template <typename T, bool ConstField>
  template <bool ConstIter>
  class NFieldMap<T, ConstField>::Iterator {
   public:
    using NFieldMap_t =
        std::conditional_t<ConstIter, const NFieldMap, NFieldMap>;
    using value_type =
        typename NFieldMap<T, ConstField>::template value_type<ConstIter>;
    using cvalue_type =
        typename NFieldMap<T, ConstField>::template value_type<true>;
    //! Default constructor
    Iterator() = delete;

    //! Constructor to beginning, or to end
    Iterator(NFieldMap_t & map, bool end)
        : map{map}, index{end ? map.size() : 0} {}

    //! Copy constructor
    Iterator(const Iterator & other) = delete;

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! Destructor
    virtual ~Iterator() = default;

    //! Copy assignment operator
    Iterator & operator=(const Iterator & other) = default;

    //! Move assignment operator
    Iterator & operator=(Iterator && other) = default;

    //! pre-increment
    Iterator & operator++() {
      this->index++;
      return *this;
    }
    //! dereference
    inline value_type operator*() { return this->map[this->index]; }
    //! dereference
    inline cvalue_type operator*() const { return this->map[this->index]; }
    //! equality
    inline bool operator==(const Iterator & other) const {
      return this->index == other.index;
    }
    //! inequality
    inline bool operator!=(const Iterator & other) const {
      return not(*this == other);
    }

   protected:
    //! NFieldMap being iterated over
    NFieldMap_t & map;
    //! current iteration index
    size_t index;
    //! map which is being returned per iterate
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_MAP_HH_
