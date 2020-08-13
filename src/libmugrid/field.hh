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
   *
   * To understand the interface, it is important to clarify the following
   * nomenclature:
   * - `Pixels` are the grid dimensions for global fields or a single linear
   * dimension for local fields
   * - `SubPts` specify the number of (tensor) quantities held per pixel. These
   * could for example be quadrature points.
   * - `Components` are the components of the physical tensor quantity
   * represented by the field.
   */
  class Field {
   protected:
    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this. This constructor initializes a field that does not know
     * the shape and storage order of its components.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_components number of components to store per sub-point
     * @param collection reference to the holding field collection.
     */
    Field(const std::string & unique_name, FieldCollection & collection,
          const Index_t & nb_components, const std::string & sub_div_tag,
          const Unit & unit);

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param components_shape number of components to store per quadrature
     * point
     * @param storage_oder in-memory storage order of the components
     */
    Field(const std::string & unique_name, FieldCollection & collection,
          const Shape_t & components_shape,
          const std::string & sub_div_tag, const Unit & unit);

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

    //! return the number of components stored per sub-point point
    const Index_t & get_nb_components() const;

    //! return the number of sub points per pixel
    const Index_t & get_nb_sub_pts() const;

    //! return the number of components stored per pixel
    Index_t get_nb_dof_per_pixel() const;

    //! return the number of pixels
    Index_t get_nb_pixels() const;

    /**
     * return the number of pixels that are required for the buffer. This can
     * be larger than get_nb_pixels if the buffer contains padding regions.
     */
    Index_t get_nb_buffer_pixels() const;

    /**
     * returns the number of entries held by this field. This corresponds to
     * nb_pixels × nb_sub_pts, (I.e., a scalar field and a vector field sharing
     * the the same collection and subdivision tag have the same number of entries, even though the
     * vector field has more scalar values.)
     */
    Index_t get_nb_entries() const;

    /**
     * returns the number of entries held by the buffer of this field. This
     * corresponds to nb_buffer_pixels × nb_sub_pts, (I.e., a scalar field and
     * a vector field sharing the the same collection have the same number of
     * entries, even though the vector field has more scalar values.)
     */
    Index_t get_nb_buffer_entries() const;

    /**
     * evaluate and return the shape of the data contained in a single sub-point
     * (e.g. quadrature point) (for passing the field to generic
     * multidimensional array objects such as numpy.ndarray)
     */
    Shape_t get_components_shape() const;

    /**
     * Reshape the components part of the field. The total number of degrees
     * of freedom per pixel must remain the same.
     */
    void reshape(const Shape_t & components_shape);

    /**
     * Reshape component and sub-point parts of the field. The total number of
     * degrees of freedom per pixel must remain the same.
     */
    void reshape(const Shape_t & components_shape,
                 const std::string & sub_div_tag);

    /**
     * evaluate and return the shape of the data contained in a single pixel
     * (for passing the field to generic multidimensional array objects such as
     * numpy.ndarray)
     */
    Shape_t get_sub_pt_shape(const IterUnit & iter_type) const;

    /**
     * evaluate and return the overall shape of the pixels portion of the field
     * (for passing the field to generic multidimensional array objects such as
     * numpy.ndarray)
     */
    Shape_t get_pixels_shape() const;

    /**
     * evaluate and return the overall shape of the field (for passing the
     * field to generic multidimensional array objects such as numpy.ndarray)
     */
    Shape_t get_shape(const IterUnit & iter_type) const;

    /**
     * evaluate and return the overall strides field (for passing the field to
     * generic multidimensional array objects such as numpy.ndarray). The
     * multiplier can be used e.g., if strides are needed in bytes, rather than
     * in pointer offsets.
     */
    virtual Shape_t get_strides(const IterUnit & iter_type,
                                Index_t element_size = 1) const;

    /**
     * Return the storage order
     */
    virtual StorageOrder get_storage_order() const;

    /**
     * evaluate and return the number of components in an iterate when iterating
     * over this field
     */
    Index_t get_stride(const IterUnit & iter_type) const;

    //! check whether two fields have the same memory layout
    bool has_same_memory_layout(const Field & other) const;

    /**
     * evaluate and return the number of rows of a default iterate over this
     * field. Warning, this function does no sanity checks at all. It is assumed
     * that the user called `get_stride` before, that all checks have been
     * performed there, and that rechecking would be a waste of time)
     */
    Index_t get_default_nb_rows(const IterUnit & iter_type) const;

    /**
     * evaluate and return the number of cols of a default iterate over this
     * field. Warning, this function does no sanity checks at all. It is assumed
     * that the user called `get_stride` before, that all checks have been
     * performed there, and that rechecking would be a waste of time)
     */
    Index_t get_default_nb_cols(const IterUnit & iter_type) const;

    /**
     * return the type information of the stored scalar (for compatibility
     * checking)
     */
    virtual const std::type_info & get_stored_typeid() const = 0;

    //! number of entries in the field (= nb_pixel × nb_sub_pts)
    Index_t get_current_nb_entries() const;

    //! size of the internal buffer including the pad region (in scalars)
    virtual size_t get_buffer_size() const = 0;

    /**
     * add a pad region to the end of the field buffer; required for using this
     * as e.g. an FFT workspace
     */
    virtual void set_pad_size(const size_t & pad_size_) = 0;

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

    //! check wether the number of pixel sub-divisions has been set
    bool has_nb_sub_pts() const;

    //! returns a const ref to the field's pixel sub-division type
    const std::string & get_sub_division_tag() const;

   protected:
    //! gives field collections the ability to resize() fields
    friend FieldCollection;

    //! sets the number of sub points per pixel
    void set_nb_sub_pts(const Index_t & nb_quad_pts_per_pixel);

    /**
     * evaluate and return the strides of the sub-point portion of the field
     * (for passing the field to generic multidimensional array objects such
     * as numpy.ndarray)
     */
    Shape_t get_components_strides(Index_t element_size = 1) const;

    /**
     * evaluate and return the strides of the pixels
     * portion of the field (for passing the field to generic multidimensional
     * array objects such as numpy.ndarray)
     */
    Shape_t get_sub_pt_strides(const IterUnit & iter_type,
                               Index_t element_size = 1) const;

    /**
     * evaluate and return the overall strides of the pixels portion of the
     * field (for passing the field to generic multidimensional array objects
     * such as numpy.ndarray)
     */
    Shape_t get_pixels_strides(Index_t element_size = 1) const;

    /**
     * maintains a tally of the current size, as it cannot be reliably
     * determined from `values` alone.
     */
    Index_t current_nb_entries{};

    //! resizes the field to the given size
    virtual void resize() = 0;

    const std::string name;  //!< the field's unique name

    //! reference to the collection this field belongs to
    FieldCollection & collection;

    /**
     * number of components stored per sub-point (e.g., 3 for a
     * three-dimensional vector, or 9 for a three-dimensional second-rank
     * tensor)
     */
    Index_t nb_components;

    /**
     * shape of the data stored per sub-point (e.g., 3, 3 for a
     * three-dimensional second-rank tensor)
     */
    Shape_t components_shape;

    //! size of padding region at end of buffer
    size_t pad_size{};

    /**
     * number of pixel subdivisions. Will depend on sub_division. This value
     * depends on the field collection and might or might not exist at
     * construction time (it would be `muGrid::Unknown` if not yes set
     */
    Index_t nb_sub_pts;

    /**
     * Pixel subdivision kind (determines how many datapoints to store per
     * pixel)
     */
    std::string sub_division_tag;

    //! Physical unit of the values stored in this field
    Unit unit;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_HH_
