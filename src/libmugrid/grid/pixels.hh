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
         *
         * Iteration is provided through three explicit methods:
         * - `indices()`: iterate over linear indices only (most efficient)
         * - `coordinates()`: iterate over grid coordinates
         * - `enumerate()`: iterate over (index, coordinate) tuples
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
             * Base iterator class storing only a linear index.
             * Coordinates are computed on-demand when needed.
             */
            class iterator {
               public:
                using difference_type = std::ptrdiff_t;
                using iterator_category = std::forward_iterator_tag;

                //! constructor
                iterator(const Pixels & pixels, Size_t index)
                    : pixels{pixels}, index{index} {}

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

                //! pre-increment
                iterator & operator++() {
                    ++this->index;
                    return *this;
                }

                //! inequality
                bool operator!=(const iterator & other) const {
                    return this->index != other.index;
                }

                //! equality
                bool operator==(const iterator & other) const {
                    return this->index == other.index;
                }

                //! get the current linear index
                Size_t get_index() const { return this->index; }

                //! get the current coordinate (computed on demand)
                DynGridIndex get_coord() const {
                    return this->pixels.get_coord(this->index);
                }

               protected:
                const Pixels & pixels;  //!< ref to pixels in cell
                Size_t index;           //!< current linear index
            };

            /**
             * Range class for index-only iteration.
             * Most efficient iteration mode - just returns linear indices.
             */
            class Indices final {
               public:
                //! Default constructor
                Indices() = delete;

                //! Constructor
                explicit Indices(const Pixels & pixels) : pixels{pixels} {}

                //! Copy constructor
                Indices(const Indices & other) = default;

                //! Move constructor
                Indices(Indices && other) = default;

                //! Destructor
                ~Indices() = default;

                //! Copy assignment operator
                Indices & operator=(const Indices & other) = delete;

                //! Move assignment operator
                Indices & operator=(Indices && other) = delete;

                /**
                 * Iterator that dereferences to linear index only.
                 */
                class iterator final : public Pixels::iterator {
                   public:
                    using Parent = Pixels::iterator;
                    using value_type = Size_t;
                    using pointer = value_type *;
                    using Parent::Parent;

                    //! dereferencing returns the linear index
                    value_type operator*() const {
                        return this->index;
                    }
                };

                //! stl conformance
                iterator begin() const { return iterator{this->pixels, 0}; }

                //! stl conformance
                iterator end() const {
                    return iterator{this->pixels, this->pixels.size()};
                }

                //! stl conformance
                size_t size() const { return this->pixels.size(); }

               protected:
                const Pixels & pixels;
            };

            /**
             * Range class for coordinate iteration.
             * Coordinates are computed on-demand during dereference.
             */
            class Coordinates final {
               public:
                //! Default constructor
                Coordinates() = delete;

                //! Constructor
                explicit Coordinates(const Pixels & pixels) : pixels{pixels} {}

                //! Copy constructor
                Coordinates(const Coordinates & other) = default;

                //! Move constructor
                Coordinates(Coordinates && other) = default;

                //! Destructor
                ~Coordinates() = default;

                //! Copy assignment operator
                Coordinates & operator=(const Coordinates & other) = delete;

                //! Move assignment operator
                Coordinates & operator=(Coordinates && other) = delete;

                /**
                 * Iterator that dereferences to grid coordinate.
                 */
                class iterator final : public Pixels::iterator {
                   public:
                    using Parent = Pixels::iterator;
                    using value_type = DynGridIndex;
                    using pointer = value_type *;
                    using Parent::Parent;

                    //! dereferencing returns the coordinate
                    value_type operator*() const {
                        return this->Parent::get_coord();
                    }
                };

                //! stl conformance
                iterator begin() const { return iterator{this->pixels, 0}; }

                //! stl conformance
                iterator end() const {
                    return iterator{this->pixels, this->pixels.size()};
                }

                //! stl conformance
                size_t size() const { return this->pixels.size(); }

               protected:
                const Pixels & pixels;
            };

            /**
             * Range class for enumerated iteration.
             * Returns tuples of (index, coordinate).
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
                ~Enumerator() = default;

                //! Copy assignment operator
                Enumerator & operator=(const Enumerator & other) = delete;

                //! Move assignment operator
                Enumerator & operator=(Enumerator && other) = delete;

                /**
                 * Iterator that dereferences to (index, coordinate) tuple.
                 */
                class iterator final : public Pixels::iterator {
                   public:
                    using Parent = Pixels::iterator;
                    using value_type = std::tuple<Index_t, DynGridIndex>;
                    using pointer = value_type *;
                    using Parent::Parent;

                    //! dereferencing returns tuple of index and coordinate
                    value_type operator*() const {
                        return value_type{
                            static_cast<Index_t>(this->index),
                            this->Parent::get_coord()};
                    }
                };

                //! stl conformance
                iterator begin() const { return iterator{this->pixels, 0}; }

                //! stl conformance
                iterator end() const {
                    return iterator{this->pixels, this->pixels.size()};
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
             * Iterate over linear indices only. Most efficient iteration mode.
             *
             * Example:
             *   for (auto index : pixels.indices()) {
             *       data[index] = ...;
             *   }
             */
            Indices indices() const { return Indices(*this); }

            /**
             * Iterate over grid coordinates.
             *
             * Example:
             *   for (auto && coord : pixels.coordinates()) {
             *       // coord is DynGridIndex
             *   }
             */
            Coordinates coordinates() const { return Coordinates(*this); }

            /**
             * Iterate over tuples of (index, coordinate).
             * Useful in parallel problems where simple enumeration would be
             * incorrect.
             *
             * Example:
             *   for (auto && [index, coord] : pixels.enumerate()) {
             *       data[index] = f(coord);
             *   }
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
