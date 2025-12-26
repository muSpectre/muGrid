/**
 * @file   gradient_operator_default.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   23 Jun 2020
 *
 * @brief  Implementation of member functions for the default gradient operator
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "operators/convolution_operator.hh"
#include "core/types.hh"
#include "collection/field_collection_global.hh"
#include "grid/index_ops.hh"
#include "util/math.hh"
#include "grid/iterators.hh"
#include "core/exception.hh"

#include <sstream>
#include <cmath>

namespace muGrid {

    /* ---------------------------------------------------------------------- */
    ConvolutionOperator::ConvolutionOperator(
        const Shape_t & pixel_offset, std::span<const Real> pixel_operator,
        const Shape_t & conv_pts_shape, const Index_t & nb_pixelnodal_pts,
        const Index_t & nb_quad_pts, const Index_t & nb_operators)
        : Parent{}, pixel_offset{pixel_offset}, pixel_operator{pixel_operator.begin(), pixel_operator.end()},
          conv_pts_shape{conv_pts_shape}, nb_pixelnodal_pts{nb_pixelnodal_pts},
          nb_quad_pts{nb_quad_pts}, nb_operators{nb_operators},
          spatial_dim{static_cast<Index_t>(conv_pts_shape.size())},
          nb_conv_pts{get_nb_from_shape(conv_pts_shape)} {
        // Check the dimension of the pixel operator
        if (pixel_operator.size() != this->nb_operators * this->nb_quad_pts * this->nb_pixelnodal_pts * this->nb_conv_pts) {
            std::stringstream err_msg{};
            err_msg << "Size mismatch: Expected the operator has "
                    << this->nb_operators * this->nb_quad_pts * this->nb_pixelnodal_pts * this->nb_conv_pts
                    << " entries. but received an operator with "
                    << pixel_operator.size() << " entries";
            throw RuntimeError{err_msg.str()};
        }
    }

    /* ---------------------------------------------------------------------- */
    const GlobalFieldCollection& ConvolutionOperator::validate_fields(
        const TypedFieldBase<Real> &nodal_field,
        const TypedFieldBase<Real> &quad_field,
        bool is_transpose) const {
        // Both fields must be from the same field collection to ensure
        // compatible internal structure for pixel mapping
        if (&nodal_field.get_collection() != &quad_field.get_collection()) {
            throw RuntimeError{
                "Field collection mismatch: nodal_field and "
                "quadrature_point_field must be from the same FieldCollection"};
        }

        // Get the collection object
        const auto & collection{dynamic_cast<const GlobalFieldCollection &>(
            quad_field.get_collection())};

        // Check that fields are global
        if (collection.get_domain() !=
            FieldCollection::ValidityDomain::Global) {
            throw RuntimeError{
                "Field type error: nodal_field and quadrature_point_field "
                "must be a global field (registered in a global FieldCollection)"};
        }

        // Check that fields have the same spatial dimensions as operator
        if (collection.get_spatial_dim() != this->spatial_dim) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: nodal_field and "
                       "quadrature_point_field are defined in "
                    << collection.get_spatial_dim()
                    << "D space, but this convolution operator is defined in "
                    << this->spatial_dim << "D space";
            throw RuntimeError{err_msg.str()};
        }

        // Ghost requirements depend on operation direction:
        // - Apply reads at (p + s), needs ghosts based on positive stencil offsets
        // - Transpose reads at (p - s), needs ghosts based on negative stencil offsets
        // For transpose, the left/right requirements are swapped.
        const auto & nb_ghosts_left{collection.get_nb_ghosts_left()};
        const auto & nb_ghosts_right{collection.get_nb_ghosts_right()};

        // Base requirements for apply operation
        const auto apply_min_left{DynGridIndex(this->spatial_dim, 0) -
                                  DynGridIndex(this->pixel_offset)};
        const auto apply_min_right{DynGridIndex(this->conv_pts_shape) -
                                   DynGridIndex(this->spatial_dim, 1) +
                                   DynGridIndex(this->pixel_offset)};

        // For transpose, swap left/right requirements
        const auto min_ghosts_left{is_transpose ? apply_min_right : apply_min_left};
        const auto min_ghosts_right{is_transpose ? apply_min_left : apply_min_right};

        for (Index_t direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_left[direction] < min_ghosts_left[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the " << (is_transpose ? "transpose" : "apply")
                    << " operation expects a minimum of "
                    << min_ghosts_left[direction]
                    << " cells on the left, but the provided fields have only "
                    << nb_ghosts_left[direction] << " ghosts on the left.";
                throw RuntimeError{err_msg.str()};
            }
        }

        for (Index_t direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_right[direction] < min_ghosts_right[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the " << (is_transpose ? "transpose" : "apply")
                    << " operation expects a minimum of "
                    << min_ghosts_right[direction]
                    << " cells on the right, but the provided fields have only "
                    << nb_ghosts_right[direction] << " ghosts on the right.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // number of components in the fields
        Index_t nb_nodal_components{nodal_field.get_nb_components()};
        Index_t nb_quad_components{quad_field.get_nb_components()};

        // check if they match
        if (nb_quad_components != this->nb_operators * nb_nodal_components) {
            std::stringstream err_msg{};
            err_msg
                << "Size mismatch: Expected a quadrature field with "
                << this->nb_operators * nb_nodal_components << " components ("
                << this->nb_operators << " operators × " << nb_nodal_components
                << " components in the nodal field) but received a field with "
                << nb_quad_components << " components.";
            throw RuntimeError{err_msg.str()};
        }

        return collection;
    }

    /* ---------------------------------------------------------------------- */
    const GlobalFieldCollection& ConvolutionOperator::validate_fields_generic(
        const Field &nodal_field,
        const Field &quad_field,
        bool is_transpose) const {
        // Both fields must be from the same field collection to ensure
        // compatible internal structure for pixel mapping
        if (&nodal_field.get_collection() != &quad_field.get_collection()) {
            throw RuntimeError{
                "Field collection mismatch: nodal_field and "
                "quadrature_point_field must be from the same FieldCollection"};
        }

        // Get the collection object
        const auto & collection{dynamic_cast<const GlobalFieldCollection &>(
            quad_field.get_collection())};

        // Check that fields are global
        if (collection.get_domain() !=
            FieldCollection::ValidityDomain::Global) {
            throw RuntimeError{
                "Field type error: nodal_field and quadrature_point_field "
                "must be a global field (registered in a global FieldCollection)"};
        }

        // Check that fields have the same spatial dimensions as operator
        if (collection.get_spatial_dim() != this->spatial_dim) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: nodal_field and "
                       "quadrature_point_field are defined in "
                    << collection.get_spatial_dim()
                    << "D space, but this convolution operator is defined in "
                    << this->spatial_dim << "D space";
            throw RuntimeError{err_msg.str()};
        }

        // Ghost requirements depend on operation direction
        const auto & nb_ghosts_left{collection.get_nb_ghosts_left()};
        const auto & nb_ghosts_right{collection.get_nb_ghosts_right()};

        // Base requirements for apply operation
        const auto apply_min_left{DynGridIndex(this->spatial_dim, 0) -
                                  DynGridIndex(this->pixel_offset)};
        const auto apply_min_right{DynGridIndex(this->conv_pts_shape) -
                                   DynGridIndex(this->spatial_dim, 1) +
                                   DynGridIndex(this->pixel_offset)};

        // For transpose, swap left/right requirements
        const auto min_ghosts_left{is_transpose ? apply_min_right : apply_min_left};
        const auto min_ghosts_right{is_transpose ? apply_min_left : apply_min_right};

        for (Index_t direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_left[direction] < min_ghosts_left[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the " << (is_transpose ? "transpose" : "apply")
                    << " operation expects a minimum of "
                    << min_ghosts_left[direction]
                    << " cells on the left, but the provided fields have only "
                    << nb_ghosts_left[direction] << " ghosts on the left.";
                throw RuntimeError{err_msg.str()};
            }
        }

        for (Index_t direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_right[direction] < min_ghosts_right[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the " << (is_transpose ? "transpose" : "apply")
                    << " operation expects a minimum of "
                    << min_ghosts_right[direction]
                    << " cells on the right, but the provided fields have only "
                    << nb_ghosts_right[direction] << " ghosts on the right.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // number of components in the fields
        Index_t nb_nodal_components{nodal_field.get_nb_components()};
        Index_t nb_quad_components{quad_field.get_nb_components()};

        // check if they match
        if (nb_quad_components != this->nb_operators * nb_nodal_components) {
            std::stringstream err_msg{};
            err_msg
                << "Size mismatch: Expected a quadrature field with "
                << this->nb_operators * nb_nodal_components << " components ("
                << this->nb_operators << " operators × " << nb_nodal_components
                << " components in the nodal field) but received a field with "
                << nb_quad_components << " components.";
            throw RuntimeError{err_msg.str()};
        }

        return collection;
    }

    /* ---------------------------------------------------------------------- */
    template<StorageOrder storage_order>
    GridTraversalParams ConvolutionOperator::compute_traversal_params(
        const GlobalFieldCollection& collection,
        Index_t nb_nodal_components,
        Index_t nb_quad_components) const {
        GridTraversalParams params;

        // Get shape of the pixels without ghosts (pad to 3D)
        auto nb_pixels_without_ghosts = pad_shape_to_3d(
            collection.get_pixels_shape_without_ghosts());
        params.nx = nb_pixels_without_ghosts[0];
        params.ny = nb_pixels_without_ghosts[1];
        params.nz = nb_pixels_without_ghosts[2];
        params.total_pixels = params.nx * params.ny * params.nz;

        // Elements per pixel - used for base offset calculation in kernel
        // For AoS: all components/subpts are interleaved per pixel
        // For SoA: each plane has separate pixels, so elems_per_pixel = 1
        if constexpr (storage_order == StorageOrder::ArrayOfStructures) {
            params.nodal_elems_per_pixel = nb_nodal_components *
                                           this->nb_pixelnodal_pts;
            params.quad_elems_per_pixel = nb_quad_components * this->nb_quad_pts;
        } else {
            // SoA: base offset is just the pixel index
            params.nodal_elems_per_pixel = 1;
            params.quad_elems_per_pixel = 1;
        }

        // Starting pixel index (skip ghosts)
        params.start_pixel_index = collection.get_pixels_index_diff();

        // Ghost counts (pad to 3D)
        auto ghosts_count = pad_shape_to_3d(
            collection.get_nb_ghosts_left() + collection.get_nb_ghosts_right(),
            0);

        // Row width including ghosts
        params.row_width = params.nx + ghosts_count[0];

        // Total buffer pixels (including ghosts)
        Index_t buffer_nx = params.nx + ghosts_count[0];
        Index_t buffer_ny = params.ny + ghosts_count[1];
        Index_t buffer_nz = params.nz + ghosts_count[2];
        Index_t total_buffer_pixels = buffer_nx * buffer_ny * buffer_nz;

        // Total elements in buffers (for SoA indexing)
        params.total_nodal_elements = total_buffer_pixels *
                                      this->nb_pixelnodal_pts;
        params.total_quad_elements = total_buffer_pixels *
                                     this->nb_quad_pts * this->nb_operators;

        // Strides depend on storage order
        if constexpr (storage_order == StorageOrder::ArrayOfStructures) {
            // AoS: components are consecutive within each pixel
            // offset = pixel_index * elems_per_pixel + elem_within_pixel
            params.nodal_stride_x = params.nodal_elems_per_pixel;
            params.nodal_stride_y = buffer_nx * params.nodal_elems_per_pixel;
            params.nodal_stride_z = params.nodal_stride_y * buffer_ny;

            params.quad_stride_x = params.quad_elems_per_pixel;
            params.quad_stride_y = buffer_nx * params.quad_elems_per_pixel;
            params.quad_stride_z = params.quad_stride_y * buffer_ny;
        } else {
            // SoA: pixels are consecutive within each component/subpoint plane
            // offset = comp * (total_pixels * subpts) + subpt * total_pixels + pixel_index
            // The strides are just pixel strides (1, buffer_nx, buffer_nx*buffer_ny)
            // because component/subpoint offsets are encoded in the sparse operator indices
            params.nodal_stride_x = 1;
            params.nodal_stride_y = buffer_nx;
            params.nodal_stride_z = buffer_nx * buffer_ny;

            params.quad_stride_x = 1;
            params.quad_stride_y = buffer_nx;
            params.quad_stride_z = buffer_nx * buffer_ny;
        }

        return params;
    }

    // Explicit template instantiations
    template GridTraversalParams ConvolutionOperator::compute_traversal_params<
        StorageOrder::ArrayOfStructures>(
        const GlobalFieldCollection&, Index_t, Index_t) const;
    template GridTraversalParams ConvolutionOperator::compute_traversal_params<
        StorageOrder::StructureOfArrays>(
        const GlobalFieldCollection&, Index_t, Index_t) const;

    /* ---------------------------------------------------------------------- */
    const SparseOperatorSoA<HostSpace>&
    ConvolutionOperator::get_apply_operator(
        const DynGridIndex & nb_grid_pts,
        const Index_t nb_nodal_components) const {
        // Check if we have a cached operator with matching parameters
        SparseOperatorCacheKey key{nb_grid_pts, nb_nodal_components};
        if (this->cached_apply_op.has_value() &&
            this->cached_key.has_value() &&
            this->cached_key.value() == key) {
            return this->cached_apply_op.value();
        }

        // Cache invalidated - clear both operators and rebuild
        // Use HostSpace::storage_order (ArrayOfStructures) for host operators
        this->cached_apply_op = this->create_apply_operator<HostSpace::storage_order>(
            nb_grid_pts, nb_nodal_components);
        this->cached_transpose_op.reset();
        this->cached_key = key;
        return this->cached_apply_op.value();
    }

    /* ---------------------------------------------------------------------- */
    const SparseOperatorSoA<HostSpace>&
    ConvolutionOperator::get_transpose_operator(
        const DynGridIndex & nb_grid_pts,
        const Index_t nb_nodal_components) const {
        // Check if we have a cached operator with matching parameters
        SparseOperatorCacheKey key{nb_grid_pts, nb_nodal_components};
        if (this->cached_transpose_op.has_value() &&
            this->cached_key.has_value() &&
            this->cached_key.value() == key) {
            return this->cached_transpose_op.value();
        }

        // Cache invalidated - clear both operators and rebuild
        // Use HostSpace::storage_order (ArrayOfStructures) for host operators
        this->cached_transpose_op = this->create_transpose_operator<HostSpace::storage_order>(
            nb_grid_pts, nb_nodal_components);
        this->cached_apply_op.reset();
        this->cached_key = key;
        return this->cached_transpose_op.value();
    }

    /* ---------------------------------------------------------------------- */
    template<StorageOrder storage_order>
    SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_apply_operator(
        const DynGridIndex & nb_grid_pts,
        const Index_t nb_nodal_components) const {
        // Helpers for conversion between col-major index and coordinate
        const CcoordOps::Pixels kernel_pixels{DynGridIndex(this->conv_pts_shape),
                                              DynGridIndex(this->pixel_offset)};
        const CcoordOps::Pixels grid_pixels{nb_grid_pts};

        // For SoA storage order, we need total element counts
        Index_t total_pixels = 1;
        for (auto dim : nb_grid_pts) {
            total_pixels *= dim;
        }
        const Index_t total_nodal_elements = total_pixels *
                                             this->nb_pixelnodal_pts;
        const Index_t total_quad_elements = total_pixels *
                                            this->nb_quad_pts * this->nb_operators;

        // First pass: count non-zero entries
        Index_t nnz = 0;
        for (const Real& value: this->pixel_operator) {
            if (std::abs(value) > this->zero_tolerance) {
                nnz += nb_nodal_components;
            }
        }

        // Allocate SoA structure
        SparseOperatorSoA<HostSpace> sparse_op;
        sparse_op.size = nnz;
        sparse_op.quad_indices.resize(nnz);
        sparse_op.nodal_indices.resize(nnz);
        sparse_op.values.resize(nnz);

        // Second pass: fill the arrays
        // Row-major order groups entries by quad index, providing write
        // locality for apply_increment (scatter to quad_data on GPU).
        Index_t entry_idx = 0;
        // The indices on quadrature field (to write) come first
        for (Index_t i_quad=0; i_quad < this->nb_quad_pts; ++i_quad) {
            for (Index_t i_operator=0; i_operator < this->nb_operators; ++i_operator) {
                // The indices on nodal field (to read)
                for (Index_t i_stencil=0; i_stencil < this->nb_conv_pts; ++i_stencil) {
                    for (Index_t i_node=0; i_node < this->nb_pixelnodal_pts; ++i_node) {
                        // Get the entry via col-major index
                        const auto op_index{((i_stencil * this->nb_pixelnodal_pts + i_node) * this-> nb_quad_pts + i_quad)
                                             * this->nb_operators + i_operator};
                        const auto op_value{this->pixel_operator[op_index]};
                        // If this is non-zero
                        if(std::abs(op_value) > this->zero_tolerance) {
                            // For each component
                            for (Index_t i_component=0; i_component < nb_nodal_components; ++i_component) {
                                // Convert from stencil index to pixel (coordinate) offset
                                const auto stencil_offset{kernel_pixels.get_coord(i_stencil)};
                                // Then to pixel count on the field
                                const auto pixel_count{grid_pixels.get_index(stencil_offset)};

                                Index_t index_diff_nodal, index_diff_quad;
                                if constexpr (storage_order == StorageOrder::ArrayOfStructures) {
                                    // AoS: (pixel_offset * pts + pt) * comps + comp
                                    index_diff_nodal = (pixel_count * this->nb_pixelnodal_pts + i_node)
                                                       * nb_nodal_components + i_component;
                                    index_diff_quad = (i_quad * this->nb_operators + i_operator)
                                                      * nb_nodal_components + i_component;
                                } else {
                                    // SoA: pixels are consecutive within each component/subpoint plane
                                    // Layout: [comp0_op0_quad0_pixels..., comp0_op0_quad1_pixels..., ...]
                                    // index = comp * total_quad_elements
                                    //       + op * (total_pixels * nb_quad_pts)
                                    //       + quad * total_pixels
                                    //       + pixel_offset (added by kernel)
                                    index_diff_nodal = i_component * total_nodal_elements
                                                       + pixel_count * this->nb_pixelnodal_pts + i_node;
                                    index_diff_quad = i_component * total_quad_elements
                                                      + i_operator * (total_pixels * this->nb_quad_pts)
                                                      + i_quad * total_pixels;
                                }

                                // Add the entry (use [] for Array access)
                                sparse_op.quad_indices[entry_idx] = index_diff_quad;
                                sparse_op.nodal_indices[entry_idx] = index_diff_nodal;
                                sparse_op.values[entry_idx] = op_value;
                                ++entry_idx;
                            }
                        }
                    }
                }
            }
        }
        return sparse_op;
    }

    // Explicit template instantiations for create_apply_operator
    template SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_apply_operator<StorageOrder::ArrayOfStructures>(
        const DynGridIndex&, Index_t) const;
    template SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_apply_operator<StorageOrder::StructureOfArrays>(
        const DynGridIndex&, Index_t) const;

    /* ---------------------------------------------------------------------- */
    template<StorageOrder storage_order>
    SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_transpose_operator(
        const DynGridIndex & nb_grid_pts,
        const Index_t nb_nodal_components) const {
        // Helpers for conversion between index and coordinates
        const CcoordOps::Pixels kernel_pixels{DynGridIndex(this->conv_pts_shape),
                                              DynGridIndex(this->pixel_offset)};
        const CcoordOps::Pixels grid_pixels{nb_grid_pts};

        // For SoA storage order, we need total element counts
        Index_t total_pixels = 1;
        for (auto dim : nb_grid_pts) {
            total_pixels *= dim;
        }
        const Index_t total_nodal_elements = total_pixels *
                                             this->nb_pixelnodal_pts;
        const Index_t total_quad_elements = total_pixels *
                                            this->nb_quad_pts * this->nb_operators;

        // First pass: count non-zero entries
        Index_t nnz = 0;
        for (const Real& value: this->pixel_operator) {
            if (std::abs(value) > this->zero_tolerance) {
                nnz += nb_nodal_components;
            }
        }

        // Allocate SoA structure
        SparseOperatorSoA<HostSpace> sparse_op;
        sparse_op.size = nnz;
        sparse_op.quad_indices.resize(nnz);
        sparse_op.nodal_indices.resize(nnz);
        sparse_op.values.resize(nnz);

        // Second pass: fill the arrays
        // Column-major order groups entries by nodal index, providing write
        // locality for transpose_increment (scatter to nodal_data on GPU).
        Index_t entry_idx = 0;
        // The indices on nodal field (to write) come first
        for (Index_t i_node=0; i_node < this->nb_pixelnodal_pts; ++i_node) {
            // The indices on quadrature field (to read)
            for (Index_t i_stencil=0; i_stencil < this->nb_conv_pts; ++i_stencil) {
                for (Index_t i_quad=0; i_quad < this->nb_quad_pts; ++i_quad) {
                    for (Index_t i_operator=0; i_operator < this->nb_operators; ++i_operator) {
                        // Get the entry via col-major index
                        const auto op_index{((i_stencil * this->nb_pixelnodal_pts + i_node) * this-> nb_quad_pts + i_quad)
                                             * this->nb_operators + i_operator};
                        const auto op_value{this->pixel_operator[op_index]};
                        // If this is non-zero
                        if(std::abs(op_value) > this->zero_tolerance) {
                            // For each component
                            for (Index_t i_component=0; i_component < nb_nodal_components; ++i_component) {
                                // Convert from stencil index to coordinate offset
                                const auto stencil_offset{kernel_pixels.get_coord(i_stencil)};
                                // Then to pixel count on the field
                                const auto pixel_count{grid_pixels.get_index(stencil_offset)};

                                Index_t index_diff_nodal, index_diff_quad;
                                if constexpr (storage_order == StorageOrder::ArrayOfStructures) {
                                    // AoS: (pixel_offset * pts + pt) * comps + comp
                                    // NOTE: pixel_count is negated for transpose (inverse mapping)
                                    index_diff_quad = ((-pixel_count * this->nb_quad_pts + i_quad) *
                                                       this->nb_operators + i_operator) *
                                                      nb_nodal_components + i_component;
                                    index_diff_nodal = i_node * nb_nodal_components + i_component;
                                } else {
                                    // SoA: pixels are consecutive within each component/subpoint plane
                                    // NOTE: pixel_count is negated for transpose (inverse mapping)
                                    // We read from quad at the stencil offset position
                                    index_diff_quad = i_component * total_quad_elements
                                                      + i_operator * (total_pixels * this->nb_quad_pts)
                                                      + i_quad * total_pixels
                                                      - pixel_count;
                                    index_diff_nodal = i_component * total_nodal_elements + i_node;
                                }

                                // Add the entry (use [] for Array access)
                                sparse_op.quad_indices[entry_idx] = index_diff_quad;
                                sparse_op.nodal_indices[entry_idx] = index_diff_nodal;
                                sparse_op.values[entry_idx] = op_value;
                                ++entry_idx;
                            }
                        }
                    }
                }
            }
        }
        return sparse_op;
    }

    // Explicit template instantiations for create_transpose_operator
    template SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_transpose_operator<StorageOrder::ArrayOfStructures>(
        const DynGridIndex&, Index_t) const;
    template SparseOperatorSoA<HostSpace>
    ConvolutionOperator::create_transpose_operator<StorageOrder::StructureOfArrays>(
        const DynGridIndex&, Index_t) const;

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::clear_cache() const {
        this->cached_apply_op.reset();
        this->cached_transpose_op.reset();
        this->cached_key.reset();
        this->cached_device_apply_op.reset();
        this->cached_device_transpose_op.reset();
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply(
        const TypedFieldBase<Real> & nodal_field,
        TypedFieldBase<Real> & quadrature_point_field) const {
        quadrature_point_field.set_zero();
        this->apply_increment(nodal_field, 1., quadrature_point_field);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const {
        // Validate fields and get collection
        const auto & collection = this->validate_fields(nodal_field,
                                                        quadrature_point_field);

        // Get component counts
        const Index_t nb_nodal_components = nodal_field.get_nb_components();
        const Index_t nb_quad_components = quadrature_point_field.get_nb_components();

        // Compute traversal parameters (use HostSpace storage order)
        const auto params = this->compute_traversal_params<HostSpace::storage_order>(
            collection, nb_nodal_components, nb_quad_components);

        // Get cached sparse operator (row-major for write locality to quad_data)
        const auto & sparse_op = this->get_apply_operator(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components);

        // Get raw data pointers
        const Real* nodal_data = nodal_field.data();
        Real* quad_data = quadrature_point_field.data();

        // Use KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<HostSpace>::apply_convolution(
            nodal_data, quad_data, alpha, params, sparse_op);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose(
        const TypedFieldBase<Real> & quadrature_point_field,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        // set nodal field to zero
        nodal_field.set_zero();
        this->transpose_increment(quadrature_point_field, 1., nodal_field,
                                  weights);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        // Validate fields and get collection (is_transpose=true for correct ghost check)
        const auto & collection = this->validate_fields(nodal_field,
                                                        quadrature_point_field,
                                                        true);

        // Get component counts
        const Index_t nb_nodal_components = nodal_field.get_nb_components();
        const Index_t nb_quad_components = quadrature_point_field.get_nb_components();

        // Compute traversal parameters (use HostSpace storage order)
        const auto params = this->compute_traversal_params<HostSpace::storage_order>(
            collection, nb_nodal_components, nb_quad_components);

        // Get cached sparse operator (col-major for write locality to nodal_data)
        const auto & sparse_op = this->get_transpose_operator(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components);

        // Get raw data pointers
        Real* nodal_data = nodal_field.data();
        const Real* quad_data = quadrature_point_field.data();

        // Use KernelDispatcher for backend-agnostic kernel execution
        KernelDispatcher<HostSpace>::transpose_convolution(
            quad_data, nodal_data, alpha, params, sparse_op);
    }

    /* ---------------------------------------------------------------------- */
    const std::vector<Real> & ConvolutionOperator::get_pixel_operator() const {
        return this->pixel_operator;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_quad_pts() const {
        return this->nb_quad_pts;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_operators() const {
        return this->nb_operators;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_nodal_pts() const {
        return this->nb_pixelnodal_pts;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_spatial_dim() const {
        return this->spatial_dim;
    }

}  // namespace muGrid
