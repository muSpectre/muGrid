/**
 * @file   numpy_tools.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Dec 2019
 *
 * @brief  Convenience functionality for working with (pybind11's) numpy arrays.
 *         These are implemented header-only, in order to avoid an explicit
 *         dependency on pybind11
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
#include "field_collection_global.hh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>

namespace py = pybind11;

namespace muGrid {

  /**
   * base class for numpy related exceptions
   */
  class NumpyError : public RuntimeError {
   public:
    //! constructor
    explicit NumpyError(const std::string & what) : RuntimeError(what) {}
    //! constructor
    explicit NumpyError(const char * what) : RuntimeError(what) {}
  };

  namespace internal {

    //! convert strides into the storage order description used by muGrid
    inline std::tuple<StorageOrder, DynCcoord_t, Shape_t>
    detect_storage_order(const DynCcoord_t & nb_subdomain_grid_pts,
                         const Shape_t & components_shape,
                         Index_t nb_sub_pts,
                         const std::vector<ssize_t> & buffer_strides,
                         ssize_t element_size) {
      auto & dim{nb_subdomain_grid_pts.get_dim()};
      if (dim + components_shape.size() != buffer_strides.size()) {
        std::stringstream s;
        s << "Internal error: Strides (= " << buffer_strides
          << ") do not match expected number of dimensions (" << dim
          << " spatial dimensions and components of shape " << components_shape
          << ").";
        throw NumpyError(s.str());
      }

      // Compute strides for SubPt and Pixel iterations passed on to directly
      // to the WrappedField if the storage order of this buffer does not match
      // the storage order intrinsically supported by muGrid.
      Shape_t strides{};

      Shape_t components_strides{};
      DynCcoord_t pixels_strides{};
      for (size_t i = 0; i < components_shape.size(); ++i) {
        components_strides.push_back(buffer_strides[i]);
        strides.push_back(buffer_strides[i] / element_size);
      }
      if (components_shape.size() > 0 and nb_sub_pts == 1) {
        // Insert a token stride for the single sub-point
        strides.push_back(0);
      }
      for (int i = 0; i < dim; ++i) {
        pixels_strides.push_back(
            buffer_strides[buffer_strides.size() - dim + i]);
        strides.push_back(
            buffer_strides[buffer_strides.size() - dim + i] / element_size);
      }

      // Determine the storage order of the whole buffer. We first check the
      // order of pixels vs. sub-points storage. This is our first guess for
      // the storage order of the whole buffer.
      StorageOrder fc_storage_order{StorageOrder::Unknown};
      if (components_strides.size() > 0) {
        const auto c{std::minmax_element(components_strides.begin(),
                                         components_strides.end())};
        const auto p{std::minmax_element(pixels_strides.begin(),
                                         pixels_strides.end())};
        if (*c.first > *p.second) {
          // this may be row-major, but we need to check that buffer is
          // contiguous
          if (*c.first == nb_subdomain_grid_pts[
                              p.second - pixels_strides.begin()] *
                              (*p.second)) {
            fc_storage_order = StorageOrder::RowMajor;
          }
        } else {
          if (*c.second <= *p.first) {
            // this may be column-major, but we need to check that buffer is
            // contiguous
            if (*p.first ==
                components_shape[c.second - components_strides.begin()] *
                    (*c.second)) {
              fc_storage_order = StorageOrder::ColMajor;
            }
          }
        }

        // the components support compact strides, but we must normalize these
        // smallest entry
        Index_t smallest_stride{*std::min_element(components_strides.begin(),
                                                  components_strides.end())};
        for (auto && s : components_strides) {
          s /= smallest_stride;
        }

        // determine order of the sub-points
        if (!CcoordOps::is_buffer_contiguous(components_shape,
                                             components_strides)) {
          fc_storage_order = StorageOrder::Unknown;
        }
      }

      // determine if the storage order of the components matches the storage
      // order determined above for the pixels vs. sub-point portion
      if (components_strides.size() > 1) {
        if (fc_storage_order == StorageOrder::ColMajor) {
          // check if the whole thing is column-major
          for (size_t i = 1; i < components_strides.size() - 1; ++i) {
            if (components_strides[i + 1] < components_strides[i]) {
              fc_storage_order = StorageOrder::Unknown;
            }
          }
        } else {
          // check if the whole thing is row-major
          for (size_t i = 1; i < components_strides.size() - 1; ++i) {
            if (components_strides[i + 1] > components_strides[i]) {
              fc_storage_order = StorageOrder::Unknown;
            }
          }
        }
      }

      // the pixels generally support arbitrary strides, but we must normalize
      // to the smallest entry
      Index_t smallest_stride{*std::min_element(pixels_strides.begin(),
                                                pixels_strides.end())};
      for (auto && s : pixels_strides) {
        s /= smallest_stride;
      }

      // If we have more than one sub-pt, the field to be wrapped must match
      // the native storage order of muGrid, because otherwise there is no way
      // to switch between SubPt and Pixel iteration.
      if (nb_sub_pts > 1) {
        if (fc_storage_order == StorageOrder::Unknown) {
          throw NumpyError(
              "Cannot wrap a field with more than one sub-point that does not "
              "match the intrinsic muGrid storage order.");
        }
      }

      if (fc_storage_order == StorageOrder::ColMajor or
          fc_storage_order == StorageOrder::RowMajor) {
        // This is a buffer that matches the intrinsic storage order of muGrid.
        // We clear the SubPt and Pixel shapes because they are correctly
        // computed by the respective Field.
        strides.clear();
      }

      return std::make_tuple(fc_storage_order, pixels_strides, strides);
    }

  }  // namespace internal

  /**
   * Construct a NumpyProxy given that we only know the number of components
   * of the field. The constructor will complain if the grid dimension differs
   * but will wrap any field whose number of components match. For example,
   * a 3x3 grid with 8 components could look like this:
   *    1. (8, 3, 3)
   *    2. (2, 4, 3, 3)
   *    3. (2, 2, 2, 3, 3)
   * The method `get_components_shape` returns the shape of the component part
   * of the field in this case. For the above examples, it would return:
   *    1. (8,)
   *    2. (2, 4)
   *    3. (2, 2, 2)
   * Note that a field with a single component can be passed either with a
   * shape having leading dimension of one or without any leading dimension.
   * In the latter case, `get_component_shape` will return a vector of size 0.
   * The same applies for fields with a single quadrature point, whose
   * dimension can be omitted. In general, the shape of the field needs to
   * look like this:
   *    (component_1, component_2, sub_pt, grid_x, grid_y, grid_z)
   * where the number of components and grid indices can be arbitrary.
   */
  template <typename T, int flags = py::array::forcecast,
            class Collection_t = GlobalFieldCollection>
  class NumpyProxy {
   public:
    /**
     * Construct a NumpyProxy given that we only know the number of components
     * of the field. The constructor will complain if the grid dimension differs
     * but will wrap any field whose number of components match. For example,
     * a 3x3 grid with 8 components could look like this:
     *    1. (8, 3, 3)
     *    2. (2, 4, 3, 3)
     *    3. (2, 2, 2, 3, 3)
     * The method `get_components_shape` returns the shape of the component part
     * of the field in this case. For the above examples, it would return:
     *    1. (8,)
     *    2. (2, 4)
     *    3. (2, 2, 2)
     * Note that a field with a single component can be passed either with a
     * shape having leading dimension of one or without any leading dimension.
     * In the latter case, `get_component_shape` will return a vector of size 0.
     * The same applies for fields with a single quadrature point, whose
     * dimension can be omitted. In general, the shape of the field needs to
     * look like this:
     *    (component_1, component_2, sub_pt, grid_x, grid_y, grid_z)
     * where the number of components and grid indices can be arbitrary.
     */
    NumpyProxy(DynCcoord_t nb_subdomain_grid_pts,
               DynCcoord_t subdomain_locations, Index_t nb_dof_per_pixel,
               pybind11::array_t<T, flags> & array,
               const Unit & unit = Unit::unitless())
        : collection{}, field{}
    {
      pybind11::buffer_info buffer = array.request();

      // Sanity check 1: Are the sizes of array and field equal?
      Index_t size{std::accumulate(
                       nb_subdomain_grid_pts.begin(),
                       nb_subdomain_grid_pts.end(), 1,
                       std::multiplies<Index_t>())*nb_dof_per_pixel};
      if (size != buffer.size) {
        std::stringstream s;
        s << "The numpy array has a size of " << buffer.size << ", but the "
          << "muGrid field reports a size of " << size << " entries.";
        throw NumpyError(s.str());
      }

      // Sanity check 2: Has the array a dimension of least the spatial dim?
      Index_t dim{nb_subdomain_grid_pts.get_dim()};
      if (buffer.shape.size() < static_cast<size_t>(dim)) {
        std::stringstream s;
        s << "The numpy array has a dimension of " << buffer.shape.size()
          << ", but the muGrid field reports a pixels dimension of " << dim
          << ".";
        throw NumpyError(s.str());
      }

      // Sanity check 3: Do the last three dimensions of the array agree with
      // the size of the grid?
      if (!std::equal(nb_subdomain_grid_pts.begin(),
                      nb_subdomain_grid_pts.end(), buffer.shape.end() - dim)) {
        std::stringstream s;
        s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
          << "field reports a grid of " << nb_subdomain_grid_pts
          << " pixels. The numpy array must equal the grid size in its last "
          << "dimensions.";
        throw NumpyError(s.str());
      }

      Index_t nb_array_components{1};
      Shape_t components_shape{};
      for (auto n{buffer.shape.begin()}; n != buffer.shape.end() - dim; ++n) {
        components_shape.push_back(*n);
        nb_array_components *= *n;
      }

      // Sanity check 4: Are the number of array components identical to the
      // number of sub-point degrees of freedom of the field? (This should not
      // fail if the above tests have not failed.)
      if (nb_array_components != nb_dof_per_pixel) {
        std::stringstream s;
        s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
          << "field reports " << nb_dof_per_pixel
          << " components. The numpy array "
          << "must equal the number of components in its first dimensions.";
        throw NumpyError(s.str());
      }

      auto storage_order{internal::detect_storage_order(
          nb_subdomain_grid_pts, components_shape, 1, buffer.strides,
          buffer.itemsize)};
      auto & fc_storage_order{std::get<0>(storage_order)};
      auto & pixels_strides{std::get<1>(storage_order)};
      auto & strides{std::get<2>(storage_order)};

      FieldCollection::SubPtMap_t map{};
      map["proxy_subpt"] = 1;
      this->collection = std::make_unique<Collection_t>(
          static_cast<Index_t>(nb_subdomain_grid_pts.get_dim()),
          nb_subdomain_grid_pts, subdomain_locations, pixels_strides, map,
          fc_storage_order);
      this->field = std::make_unique<WrappedField<T>>(
          "proxy_field", *collection, components_shape,
          static_cast<size_t>(buffer.size),
          static_cast<T *>(buffer.ptr),
          "proxy_subpt", unit, strides);
    }

    /**
     * Construct a NumpyProxy given that we know the shape of the leading
     * component indices. The constructor will complain if both the grid
     * dimensions and the component dimensions differ. `get_component_shape`
     * returns exactly the shape passed to this constructor.
     *
     * In general, the shape of the field needs to look like this:
     *    (component_1, component_2, sub_pt, grid_x, grid_y, grid_z)
     * where the number of components and grid indices can be arbitrary. The
     * sub_pt dimension can be omitted if there is only a single sub-point.
     */
    NumpyProxy(DynCcoord_t nb_subdomain_grid_pts,
               DynCcoord_t subdomain_locations, Index_t nb_sub_pts,
               Shape_t components_shape,
               pybind11::array_t<T, flags> & array,
               const Unit & unit = Unit::unitless())
        : collection{}, field{}
    {
      pybind11::buffer_info buffer{array.request()};

      // Sanity check: Do the array dimensions agree and shapes agree?
      Index_t dim{nb_subdomain_grid_pts.get_dim()};
      bool shape_matches{false};
      Shape_t sub_pt_shape{components_shape};
      if (dim + components_shape.size() + 1 == buffer.shape.size()) {
        shape_matches =
            std::equal(nb_subdomain_grid_pts.begin(),
                       nb_subdomain_grid_pts.end(), buffer.shape.end() - dim) &&
            nb_sub_pts == buffer.shape[components_shape.size()] &&
            std::equal(components_shape.begin(), components_shape.end(),
                       buffer.shape.begin());
        sub_pt_shape.push_back(nb_sub_pts);
      } else if (dim + components_shape.size() == buffer.shape.size()) {
        // For a field with a single quad point, we can omit that dimension.
        shape_matches =
            std::equal(nb_subdomain_grid_pts.begin(),
                       nb_subdomain_grid_pts.end(), buffer.shape.end() - dim) &&
            nb_sub_pts == 1 &&
            std::equal(components_shape.begin(), components_shape.end(),
                       buffer.shape.begin());
      }
      if (!shape_matches) {
        std::stringstream s;
        s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
          << "field reports a grid of " << nb_subdomain_grid_pts
          << " pixels with " << nb_sub_pts << " quadrature "
          << (nb_sub_pts == 1 ? "point" : "points") << " holding a quantity of "
          << "shape " << components_shape << ".";
        throw NumpyError(s.str());
      }

      auto storage_order{internal::detect_storage_order(
          nb_subdomain_grid_pts, sub_pt_shape, nb_sub_pts, buffer.strides,
          buffer.itemsize)};
      auto & fc_storage_order{std::get<0>(storage_order)};
      auto & pixels_strides{std::get<1>(storage_order)};
      auto & sub_pt_iter_strides{std::get<2>(storage_order)};

      FieldCollection::SubPtMap_t map{};
      map["proxy_subpt"] = nb_sub_pts;
      this->collection = std::make_unique<Collection_t>(
          static_cast<Index_t>(nb_subdomain_grid_pts.get_dim()),
          nb_subdomain_grid_pts, subdomain_locations, pixels_strides, map,
          fc_storage_order);
      this->field = std::make_unique<WrappedField<T>>(
          "proxy_field", *collection, components_shape,
          static_cast<size_t>(buffer.size),
          static_cast<T *>(buffer.ptr),
          "proxy_subpt", unit, sub_pt_iter_strides);
    }

    /**
     * move constructor
     */
    NumpyProxy(NumpyProxy && other) = default;

    WrappedField<T> & get_field() { return *this->field; }

    const Shape_t get_components_shape() const {
      return this->field->get_components_shape();
    }

    Shape_t get_sub_pt_shape() const {
      return this->field->get_sub_pt_shape();
    }

   protected:
    std::unique_ptr<Collection_t> collection;
    std::unique_ptr<WrappedField<T>> field;
  };

  /* Wrap a column-major field into a numpy array, without copying the data */
  template <typename T>
  pybind11::array_t<T, pybind11::array::f_style>
  numpy_wrap(const TypedFieldBase<T> & field,
             IterUnit iter_type = IterUnit::SubPt) {
    Shape_t shape{field.get_shape(iter_type)};
    Shape_t strides{field.get_strides(iter_type, sizeof(T))};
    return pybind11::array_t<T, py::array::f_style>(
        shape, strides, field.data(), pybind11::capsule([]() {}));
  }

  /* Turn any type that can be enumerated into a tuple */
  template <typename T>
  pybind11::tuple to_tuple(T a) {
    pybind11::tuple t(a.get_dim());
    ssize_t i = 0;
    for (auto && v : a) {
      t[i] = v;
      i++;
    }
    return t;
  }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NUMPY_TOOLS_HH_
