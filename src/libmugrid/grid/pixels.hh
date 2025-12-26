/**
 * @file   grid/pixels.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  Pixels iterator class for iteration over discretisation grids
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_LIBMUGRID_GRID_PIXELS_HH_
#define SRC_LIBMUGRID_GRID_PIXELS_HH_

#include "core/types.hh"
#include "core/coordinates.hh"
#include "core/exception.hh"
#include "strides.hh"
#include "index_ops.hh"

#include <tuple>

namespace muGrid {
    namespace CcoordOps {
        /**
         * Iteration over square (or cubic) discretisation grids. Duplicates
         * capabilities of `muGrid::CcoordOps::Pixels` without needing to be
         * templated with the spatial dimension. Iteration is slower, though.
         */
        class Pixels {
           public:
            Pixels();

            //! Constructor with default strides (column-major pixel storage
            //! order)
            explicit Pixels(
                const DynGridIndex & nb_subdomain_grid_pts,
                const DynGridIndex & subdomain_locations = DynGridIndex{});

            /**
             * Constructor with custom strides (any, including partially
             * transposed pixel storage order)
             */
            Pixels(const DynGridIndex & nb_subdomain_grid_pts,
                   const DynGridIndex & subdomain_locations,
                   const DynGridIndex & strides);

            //! Constructor with default strides from statically sized coords
            template <size_t Dim>
            explicit Pixels(
                const GridIndex<Dim> & nb_subdomain_grid_pts,
                const GridIndex<Dim> & subdomain_locations = GridIndex<Dim>{});

            //! Constructor with custom strides from statically sized coords
            template <size_t Dim>
            Pixels(const GridIndex<Dim> & nb_subdomain_grid_pts,
                   const GridIndex<Dim> & subdomain_locations,
                   const GridIndex<Dim> & strides);

            //! Copy constructor
            Pixels(const Pixels & other) = default;

            //! Move constructor
            Pixels(Pixels && other) = default;

            //! Destructor
            virtual ~Pixels() = default;

            //! Copy assignment operator
            Pixels & operator=(const Pixels & other) = default;

            //! Move assignment operator
            Pixels & operator=(Pixels && other) = default;

            //! evaluate and return the linear index corresponding to dynamic
            //! `ccoord`
            Index_t get_index(const DynGridIndex & ccoord) const {
                return get_index_from_strides(
                    this->strides, this->subdomain_locations, ccoord);
            }

            //! evaluate and return the linear index corresponding to `ccoord`
            template <size_t Dim>
            Index_t get_index(const GridIndex<Dim> & ccoord) const {
                if (this->dim != Dim) {
                    throw RuntimeError("dimension mismatch");
                }
                return get_index_from_strides(
                    this->strides.template get<Dim>(),
                    this->subdomain_locations.template get<Dim>(), ccoord);
            }

            //! return coordinates of the i-th pixel
            DynGridIndex get_coord(const Index_t & index) const {
                return get_coord_from_axes_order(
                    this->nb_subdomain_grid_pts, this->subdomain_locations,
                    this->strides, this->axes_order, index);
            }

            //! return coordinates of the i-th pixel, with zero as location
            DynGridIndex get_coord0(const Index_t & index) const {
                return get_coord0_from_axes_order(this->nb_subdomain_grid_pts,
                                                  this->strides,
                                                  this->axes_order, index);
            }

            DynGridIndex get_neighbour(const DynGridIndex & ccoord,
                                     const DynGridIndex & offset) const {
                return modulo(ccoord + offset - this->subdomain_locations,
                              this->nb_subdomain_grid_pts) +
                       this->subdomain_locations;
            }

            /**
             * Iterator class for `muSpectre::Pixels`
             */
            class iterator {
               public:
                //! stl
                using value_type = DynGridIndex;
                using const_value_type = const value_type;  //!< stl conformance
                using pointer = value_type *;               //!< stl conformance
                using difference_type = std::ptrdiff_t;     //!< stl conformance
                using iterator_category = std::forward_iterator_tag;
                //!< stl
                //!< conformance

                //! constructor
                iterator(const Pixels & pixels, Size_t index)
                    : pixels{pixels}, coord0{pixels.get_coord0(index)} {}

                //! constructor
                iterator(const Pixels & pixels, DynGridIndex coord0)
                    : pixels{pixels}, coord0{coord0} {}

                //! Default constructor
                iterator() = delete;

                //! Copy constructor
                iterator(const iterator & other) = default;

                //! Move constructor
                iterator(iterator && other) = default;

                //! Destructor
                ~iterator() = default;

                //! Copy assignment operator
                iterator & operator=(const iterator & other) = delete;

                //! Move assignment operator
                iterator & operator=(iterator && other) = delete;

                //! dereferencing
                value_type operator*() const {
                    return this->pixels.subdomain_locations + this->coord0;
                }

                //! pre-increment
                iterator & operator++() {
                    auto axis{this->pixels.axes_order[0]};
                    // Increase fastest index
                    ++this->coord0[axis];
                    // Check whether coordinate is out of bounds
                    Index_t aindex{0};
                    while (aindex < this->pixels.dim - 1 &&
                           this->coord0[axis] >=
                               this->pixels.nb_subdomain_grid_pts[axis]) {
                        this->coord0[axis] = 0;
                        // Get next fastest axis
                        axis = this->pixels.axes_order[++aindex];
                        ++this->coord0[axis];
                    }
                    return *this;
                }

                //! inequality
                bool operator!=(const iterator & other) const {
                    return this->coord0 != other.coord0;
                }

                //! equality
                bool operator==(const iterator & other) const {
                    return not(*this != other);
                }

               protected:
                const Pixels & pixels;  //!< ref to pixels in cell
                DynGridIndex coord0;      //!< coordinate of current pixel
            };

            //! stl conformance
            iterator begin() const { return iterator(*this, 0); }

            //! stl conformance
            iterator end() const {
                return ++iterator(*this, this->nb_subdomain_grid_pts - 1);
            }

            //! stl conformance
            size_t size() const {
                return get_size(this->nb_subdomain_grid_pts);
            }

            //! buffer size, including padding
            size_t buffer_size() const {
                return get_buffer_size(this->nb_subdomain_grid_pts,
                                       this->strides);
            }

            //! return spatial dimension
            Dim_t get_dim() const { return this->dim; }

            //! return the resolution of the discretisation grid in each spatial
            //! dim
            const DynGridIndex & get_nb_subdomain_grid_pts() const {
                return this->nb_subdomain_grid_pts;
            }

            /**
             * return the ccoordinates of the bottom, left, (front) pixel/voxel
             * of this processors partition of the discretisation grid. For
             * sequential calculations, this is alvays the origin
             */
            const DynGridIndex & get_subdomain_locations() const {
                return this->subdomain_locations;
            }

            //! return the strides used for iterating over the pixels
            const DynGridIndex & get_strides() const { return this->strides; }

            /**
             * enumerator class for `muSpectre::Pixels`
             */
            class Enumerator final {
               public:
                //! Default constructor
                Enumerator() = delete;

                //! Constructor
                explicit Enumerator(const Pixels & pixels) : pixels{pixels} {}

                //! Copy constructor
                Enumerator(const Enumerator & other) = default;

                //! Move constructor
                Enumerator(Enumerator && other) = default;

                //! Destructor
                virtual ~Enumerator() = default;

                //! Copy assignment operator
                Enumerator & operator=(const Enumerator & other) = delete;

                //! Move assignment operator
                Enumerator & operator=(Enumerator && other) = delete;

                /**
                 * @class iterator
                 * @brief A derived class from Pixels::iterator, used for
                 * iterating over Pixels.
                 *
                 * This class is a final class, meaning it cannot be further
                 * derived from. It provides a custom implementation of the
                 * dereference operator (*).
                 *
                 * @tparam Parent Alias for the base class Pixels::iterator.
                 *
                 * @note The using Parent::Parent; statement is a C++11 feature
                 * called "Inheriting Constructors" which means that this
                 * derived class will have the same constructors as the base
                 * class.
                 */
                class iterator final : public Pixels::iterator {
                   public:
                    using Parent = Pixels::iterator;
                    using Parent::Parent;

                    /**
                     * @brief Overloaded dereference operator (*).
                     *
                     * This function returns a tuple containing the index of the
                     * pixel and the pixel's coordinates.
                     *
                     * @return std::tuple<Index_t, Parent::value_type> A tuple
                     * containing the index of the pixel and the pixel's
                     * coordinates.
                     */
                    std::tuple<Index_t, Parent::value_type> operator*() const {
                        auto && pixel{this->Parent::operator*()};
                        return std::tuple<Index_t, Parent::value_type>{
                            this->pixels.get_index(pixel), pixel};
                    }
                };

                //! stl conformance
                iterator begin() const { return iterator{this->pixels, 0}; }

                //! stl conformance
                iterator end() const {
                    iterator it{this->pixels,
                                this->pixels.nb_subdomain_grid_pts - 1};
                    ++it;
                    return it;
                }

                //! stl conformance
                size_t size() const { return this->pixels.size(); }

                size_t buffer_size() const {
                    return this->pixels.buffer_size();
                }

               protected:
                const Pixels & pixels;
            };

            /**
             * iterates in tuples of pixel index ond coordinate. Useful in
             * parallel problems, where simple enumeration of the pixels would
             * be incorrect
             */
            Enumerator enumerate() const { return Enumerator(*this); }

           protected:
            Dim_t dim;                         //!< spatial dimension
            DynGridIndex nb_subdomain_grid_pts;  //!< nb_grid_pts of this domain
            DynGridIndex subdomain_locations;    //!< locations of this domain
            DynGridIndex strides;                //!< strides of memory layout
            DynGridIndex axes_order;             //!< order of axes
            bool contiguous;                   //!< is this a contiguous buffer?
        };
    }  // namespace CcoordOps
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_GRID_PIXELS_HH_
