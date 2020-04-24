/**
 * @file   field.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Aug 2019
 *
 * @brief  Base class for fields
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

#ifndef SRC_LIBMUGRID_FIELD_HH_
#define SRC_LIBMUGRID_FIELD_HH_

#include "exception.hh"
#include "grid_common.hh"
#include "units.hh"

#include <string>
#include <typeinfo>

namespace muGrid {

  /**
   * base class for field-related exceptions
   */
  class FieldError : public RuntimeError {
   public:
    //! constructor
    explicit FieldError(const std::string & what) : RuntimeError(what) {}
    //! constructor
    explicit FieldError(const char * what) : RuntimeError(what) {}
  };

  //! forward-declaration
  class FieldCollection;

  //! forward-declaration
  class StateField;

  /**
   * Abstract base class for all fields. A field provides storage discretising a
   * mathematical (scalar, vectorial, tensorial) (real-valued,
   * integer-valued, complex-valued) field on a fixed number of quadrature
   * points per pixel/voxel of a regular grid.
   * Fields defined on the same domains are grouped within
   * `muGrid::FieldCollection`s.
   */
  class Field {
   protected:
    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_dof_per_quad_pt number of components to store per quadrature point
     * @param collection reference to the holding field collection.
     */
    Field(const std::string & unique_name, FieldCollection & collection,
          Dim_t nb_dof_per_quad_pt, const Unit & unit);

   public:
    //! Default constructor
    Field() = delete;

    //! Copy constructor
    Field(const Field & other) = delete;

    //! Move constructor
    Field(Field && other) = default;

    //! Destructor
    virtual ~Field() = default;

    //! Copy assignment operator
    Field & operator=(const Field & other) = delete;

    //! Move assignment operator
    Field & operator=(Field && other) = delete;

    //! return the field's unique name
    const std::string & get_name() const;

    //! return a const reference to the field's collection
    FieldCollection & get_collection() const;

    //! return the number of components stored per quadrature point
    const Dim_t & get_nb_dof_per_quad_pt() const;

    //! return the number of quadrature points per pixel
    const Dim_t & get_nb_quad_pts() const;

    //! return the number of components stored per pixel
    Dim_t get_nb_dof_per_pixel() const;

    //! return the number of pixels
    Dim_t get_nb_pixels() const;

    /**
     * evaluate and return the overall shape of the field (for passing the
     * field to generic multidimensional array objects such as numpy.ndarray)
     */
    std::vector<Dim_t> get_shape(Iteration iter_type) const;

    /**
     * evaluate and return the overall shape of the pixels portion of the field
     * (for passing the field to generic multidimensional array objects such as
     * numpy.ndarray)
     */
    std::vector<Dim_t> get_pixels_shape() const;

    /**
     * evaluate and return the shape of the data contained in a single pixel or
     * quadrature point (for passing the field to generic multidimensional
     * array objects such as numpy.ndarray)
     */
    virtual std::vector<Dim_t> get_components_shape(Iteration iter_type) const;

    /**
     * evaluate and return the number of components in an iterate when iterating
     * over this field
     */
    Dim_t get_stride(Iteration iter_type) const;

    /**
     * return the type information of the stored scalar (for compatibility
     * checking)
     */
    virtual const std::type_info & get_stored_typeid() const = 0;

    //! number of entries in the field (= nb_pixel × nb_quad)
    size_t size() const;

    //! size of the internal buffer including the pad region (in scalars)
    virtual size_t buffer_size() const = 0;

    /**
     * add a pad region to the end of the field buffer; required for using this
     * as e.g. an FFT workspace
     */
    virtual void set_pad_size(size_t pad_size_) = 0;

    //! pad region size
    const size_t & get_pad_size() const;

    /**
     * initialise field to zero (do more complicated initialisations through
     * fully typed maps)
     */
    virtual void set_zero() = 0;

    /**
     * checks whether this field is registered in a global FieldCollection
     */
    bool is_global() const;

   protected:
    //! gives field collections the ability to resize() fields
    friend FieldCollection;

    /**
     * maintains a tally of the current size, as it cannot be reliably
     * determined from either `values` or `alt_values` alone.
     */
    size_t current_size{};

    //! resizes the field to the given size
    virtual void resize(size_t size) = 0;

    const std::string name;  //!< the field's unique name

    //! reference to the collection this field belongs to
    FieldCollection & collection;

    /**
     * number of components stored per quadrature point (e.g., 3 for a
     * three-dimensional vector, or 9 for a three-dimensional second-rank
     * tensor)
     */
    const Dim_t nb_dof_per_quad_pt;

    //! size of padding region at end of buffer
    size_t pad_size{};

    //! Physical unit of the values stored in this field
    Unit unit;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_HH_
