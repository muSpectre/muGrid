/**
 * @file   variable_stencil.hh
 *
 * @author Martin Ladecký <martin.ladecky@imtek.uni-freiburg.de>
 *
 * @date   16 Nov 2026
 *
 * @brief  Nodal operator with a different symmetric 3^Dim stencil at every
 *         node (the Galerkin coarse operators of the heterogeneous multigrid)
 *
 * Copyright © 2026 Martin Ladecký
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

#ifndef SRC_LIBMUGRID_OPERATORS_VARIABLE_STENCIL_HH_
#define SRC_LIBMUGRID_OPERATORS_VARIABLE_STENCIL_HH_

#include "core/types.hh"
#include "collection/field_collection_global.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "memory/gpu_runtime.hh"
#endif

#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace muGrid {

    /**
     * Storage layout of a symmetric nodal 3^Dim stencil (shared with Python,
     * muGrid/Multigrid.py, through the bindings -- there is one definition).
     *
     * The operator is y_i(n) = sum_s sum_j A(n, n+s)_ij x_j(n+s) over the
     * offsets s in {-1, 0, 1}^Dim, with nc x nc blocks. It is symmetric,
     * A(n, n+s) = A(n+s, n)^T, so only half of the blocks are stored:
     *
     * - the "positive" offsets H = { s : f(s) > f(0) }, in increasing f, with
     *   the flat index f(s) = sum_a (s_a + 1) 3^a (x fastest, as in
     *   LaplaceOperator::get_coefficients): 4 offsets in 2D, 13 in 3D. Entry
     *   k * nc^2 + i * nc + j of node n is A(n, n + s_k)_ij.
     * - the centre block A(n, n), symmetric, as its upper triangle row by row:
     *   entry |H| nc^2 + t(i, j), t(i, j) = i nc - i (i + 1) / 2 + j, i <= j.
     *
     * The block A(n, n - s_k) of a negative offset is A_k(n - s_k)^T, read at
     * the neighbour. So the apply reads the stencil field one node away on
     * either side and needs one ghost layer there, filled by
     * communicate_ghosts after the stencil was set.
     */
    namespace variable_stencil {

        //! Number of stored ("positive") offsets: (3^Dim - 1) / 2.
        constexpr Index_t nb_half_offsets(Dim_t dim) {
            return dim == 2 ? 4 : 13;
        }

        //! Scalars per node: |H| nc^2 blocks plus the centre upper triangle.
        constexpr Index_t nb_entries(Dim_t dim, Index_t nb_components) {
            return nb_half_offsets(dim) * nb_components * nb_components +
                   nb_components * (nb_components + 1) / 2;
        }

        //! The offsets H in storage order (see the namespace comment).
        template <Dim_t Dim>
        std::vector<std::array<Index_t, Dim>> half_offsets() {
            Index_t centre{0};
            Index_t power{1};
            for (Dim_t a{0}; a < Dim; ++a) {
                centre += power;
                power *= 3;
            }
            std::vector<std::array<Index_t, Dim>> offsets{};
            for (Index_t flat{centre + 1}; flat < power; ++flat) {
                std::array<Index_t, Dim> s{};
                Index_t rest{flat};
                for (Dim_t a{0}; a < Dim; ++a) {
                    s[a] = rest % 3 - 1;
                    rest /= 3;
                }
                offsets.push_back(s);
            }
            return offsets;
        }

        /**
         * Strides of one apply, in scalars. Pixel strides are per axis (the
         * third is unused in 2D, with nb_interior[2] = 1); the dof strides
         * address the components of a node (1 on the host, array of
         * structures). Pointers handed to the kernels point at the first
         * interior node.
         */
        struct Geometry {
            std::array<Index_t, 3> nb_interior{1, 1, 1};
            std::array<Index_t, 3> node_stride{0, 0, 0};
            Index_t node_stride_dof{1};
            std::array<Index_t, 3> stencil_stride{0, 0, 0};
            Index_t stencil_stride_dof{1};
            //! per stored offset s_k: scalar offset of the node n + s_k
            std::vector<Index_t> node_offset{};
            //! per stored offset s_k: scalar offset of the stencil of n + s_k
            std::vector<Index_t> stencil_offset{};
        };

        /**
         * @brief y = alpha A x (or y += alpha A x) on the interior nodes,
         *        host. `nb_components` in {1, 2, 3}.
         */
        template <typename T>
        void apply_host(const T * MUGRID_RESTRICT stencil,
                        const T * MUGRID_RESTRICT x, T * MUGRID_RESTRICT y,
                        Index_t nb_components, Index_t nb_half,
                        const Geometry & geometry, T alpha, bool increment);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! device version of apply_host (structure-of-arrays strides)
        template <typename T>
        void apply_gpu(const T * stencil, const T * x, T * y,
                       Index_t nb_components, Index_t nb_half,
                       const Geometry & geometry, T alpha, bool increment);
#endif

    }  // namespace variable_stencil

    /**
     * @class VariableStencilOperator
     * @brief Nodal-to-nodal operator with its own symmetric 3^Dim stencil of
     *        nc x nc blocks at every node.
     *
     * This is the explicit form of a Galerkin coarse operator R A P: for an
     * operator A that couples only neighbouring nodes and the multilinear
     * prolongation P, R A P couples each coarse node with its 3^Dim
     * neighbours, with coefficients that vary from node to node when A is
     * heterogeneous. The coefficients live in a field on the same collection
     * as x and y (`nb_entries` scalars per node, layout in the
     * `variable_stencil` namespace comment); half of the blocks are stored,
     * because the operator is symmetric.
     *
     * The operator object holds no data; the stencil field is passed to every
     * apply, like the material fields of IsotropicStiffnessOperator.
     *
     * Ghosts: x and the stencil field need one ghost layer on each side,
     * filled by communicate_ghosts (the stencil field once after it was set,
     * x before every apply). Only the interior nodes of y are written.
     * Cost: (2|H| + 1) nc^2 multiply-adds per node, reading nb_entries + nc
     * scalars per node.
     */
    template <Dim_t Dim>
    class VariableStencilOperator {
        static_assert(Dim == 2 || Dim == 3,
                      "VariableStencilOperator is implemented for 2D and 3D");

       public:
        static constexpr Index_t NB_HALF_OFFSETS{
            variable_stencil::nb_half_offsets(Dim)};

        explicit VariableStencilOperator(Index_t nb_components)
            : nb_components{nb_components} {
            if (nb_components < 1 || nb_components > 3) {
                std::stringstream err{};
                err << operator_name() << ": nb_components must be 1, 2 or 3, "
                    << "got " << nb_components;
                throw RuntimeError{err.str()};
            }
        }

        VariableStencilOperator() = delete;

        Index_t get_nb_components() const { return this->nb_components; }

        Index_t get_nb_entries() const {
            return variable_stencil::nb_entries(Dim, this->nb_components);
        }

        static Dim_t get_spatial_dim() { return Dim; }

        static std::vector<std::array<Index_t, Dim>> get_half_offsets() {
            return variable_stencil::half_offsets<Dim>();
        }

        //! y = A x
        void apply(const TypedFieldBase<Real> & stencil,
                   const TypedFieldBase<Real> & x,
                   TypedFieldBase<Real> & y) const {
            this->apply_impl<Real>(stencil, x, y, 1.0, false);
        }

        //! y += alpha A x
        void apply_increment(const TypedFieldBase<Real> & stencil,
                             const TypedFieldBase<Real> & x, Real alpha,
                             TypedFieldBase<Real> & y) const {
            this->apply_impl<Real>(stencil, x, y, alpha, true);
        }

        void apply(const TypedFieldBase<Real32> & stencil,
                   const TypedFieldBase<Real32> & x,
                   TypedFieldBase<Real32> & y) const {
            this->apply_impl<Real32>(stencil, x, y, 1.0f, false);
        }

        void apply_increment(const TypedFieldBase<Real32> & stencil,
                             const TypedFieldBase<Real32> & x, Real32 alpha,
                             TypedFieldBase<Real32> & y) const {
            this->apply_impl<Real32>(stencil, x, y, alpha, true);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        using DField = TypedFieldBase<Real, DefaultDeviceSpace>;
        using DField32 = TypedFieldBase<Real32, DefaultDeviceSpace>;

        void apply(const DField & stencil, const DField & x,
                   DField & y) const {
            this->apply_impl<Real>(stencil, x, y, 1.0, false);
        }

        void apply_increment(const DField & stencil, const DField & x,
                             Real alpha, DField & y) const {
            this->apply_impl<Real>(stencil, x, y, alpha, true);
        }

        void apply(const DField32 & stencil, const DField32 & x,
                   DField32 & y) const {
            this->apply_impl<Real32>(stencil, x, y, 1.0f, false);
        }

        void apply_increment(const DField32 & stencil, const DField32 & x,
                             Real32 alpha, DField32 & y) const {
            this->apply_impl<Real32>(stencil, x, y, alpha, true);
        }
#endif

       protected:
        static const char * operator_name() {
            return Dim == 2 ? "VariableStencilOperator2D"
                            : "VariableStencilOperator3D";
        }

        /**
         * Check the three fields and resolve the strides of one apply.
         * `device` selects the structure-of-arrays layout of device fields.
         */
        variable_stencil::Geometry validate(const Field & stencil,
                                            const Field & x, const Field & y,
                                            bool device) const {
            const std::string op{operator_name()};
            const auto * fc = dynamic_cast<const GlobalFieldCollection *>(
                &x.get_collection());
            if (fc == nullptr) {
                throw RuntimeError{op + ": the fields must live on a "
                                        "GlobalFieldCollection"};
            }
            if (fc->get_spatial_dim() != Dim) {
                std::stringstream err{};
                err << op << ": the fields are " << fc->get_spatial_dim()
                    << "-dimensional";
                throw RuntimeError{err.str()};
            }
            if (&stencil.get_collection() != fc ||
                &y.get_collection() != fc) {
                throw RuntimeError{op + ": stencil, x and y must live on the "
                                        "same field collection"};
            }
            if (&x == &y) {
                throw RuntimeError{op + ": x and y must be different fields"};
            }
            for (const Field * f : {&x, &y}) {
                if (f->get_nb_sub_pts() != 1 ||
                    f->get_nb_components() != this->nb_components) {
                    std::stringstream err{};
                    err << op << ": field '" << f->get_name() << "' has "
                        << f->get_nb_components() << " component(s) on "
                        << f->get_nb_sub_pts() << " sub-point(s); expected "
                        << this->nb_components
                        << " components on one nodal point per pixel";
                    throw RuntimeError{err.str()};
                }
            }
            const Index_t nb_entries{this->get_nb_entries()};
            if (stencil.get_nb_sub_pts() != 1 ||
                stencil.get_nb_components() != nb_entries) {
                std::stringstream err{};
                err << op << ": the stencil field '" << stencil.get_name()
                    << "' has " << stencil.get_nb_components()
                    << " component(s) on " << stencil.get_nb_sub_pts()
                    << " sub-point(s); expected " << nb_entries
                    << " components (nb_entries) on one point per pixel";
                throw RuntimeError{err.str()};
            }

            const auto & pixel_strides{
                fc->get_pixels_with_ghosts().get_strides()};
            const auto & with_ghosts{
                fc->get_nb_subdomain_grid_pts_with_ghosts()};
            const auto & ghosts_left{fc->get_nb_ghosts_left()};
            const auto & ghosts_right{fc->get_nb_ghosts_right()};
            const auto interior{fc->get_nb_subdomain_grid_pts_without_ghosts()};

            variable_stencil::Geometry g{};
            Index_t nb_pixels{1};
            for (Dim_t a{0}; a < Dim; ++a) {
                nb_pixels *= with_ghosts[a];
            }
            // Host fields are arrays of structures (components fastest),
            // device fields structures of arrays (one plane per component).
            const Index_t node_pixel{device ? 1 : this->nb_components};
            const Index_t stencil_pixel{device ? 1 : nb_entries};
            g.node_stride_dof = device ? nb_pixels : 1;
            g.stencil_stride_dof = device ? nb_pixels : 1;
            Index_t expected{1};
            for (Dim_t a{0}; a < Dim; ++a) {
                if (pixel_strides[a] != expected) {
                    std::stringstream err{};
                    err << op << ": the collection must use dense "
                        << "column-major pixel storage, but has pixel stride "
                        << pixel_strides[a] << " on axis " << a;
                    throw RuntimeError{err.str()};
                }
                if (ghosts_left[a] < 1 || ghosts_right[a] < 1) {
                    std::stringstream err{};
                    err << op << ": the collection needs at least one ghost "
                        << "layer on each side of axis " << a << " (has "
                        << ghosts_left[a] << " left, " << ghosts_right[a]
                        << " right)";
                    throw RuntimeError{err.str()};
                }
                g.nb_interior[a] = interior[a];
                g.node_stride[a] = expected * node_pixel;
                g.stencil_stride[a] = expected * stencil_pixel;
                expected *= with_ghosts[a];
            }
            for (const auto & s : variable_stencil::half_offsets<Dim>()) {
                Index_t node{0}, sten{0};
                for (Dim_t a{0}; a < Dim; ++a) {
                    node += s[a] * g.node_stride[a];
                    sten += s[a] * g.stencil_stride[a];
                }
                g.node_offset.push_back(node);
                g.stencil_offset.push_back(sten);
            }
            return g;
        }

        //! scalar offset of the first interior node (pixel strides of g)
        Index_t first_interior(const Field & field,
                               const std::array<Index_t, 3> & strides) const {
            const auto & fc =
                dynamic_cast<const GlobalFieldCollection &>(
                    field.get_collection());
            const auto & ghosts_left{fc.get_nb_ghosts_left()};
            Index_t offset{0};
            for (Dim_t a{0}; a < Dim; ++a) {
                offset += ghosts_left[a] * strides[a];
            }
            return offset;
        }

        template <typename T>
        void apply_impl(const TypedFieldBase<T> & stencil,
                        const TypedFieldBase<T> & x, TypedFieldBase<T> & y,
                        T alpha, bool increment) const {
            const auto g{this->validate(stencil, x, y, false)};
            const Index_t node0{this->first_interior(x, g.node_stride)};
            const Index_t sten0{
                this->first_interior(stencil, g.stencil_stride)};
            variable_stencil::apply_host<T>(
                stencil.data() + sten0, x.data() + node0, y.data() + node0,
                this->nb_components, NB_HALF_OFFSETS, g, alpha, increment);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        template <typename T>
        void apply_impl(const TypedFieldBase<T, DefaultDeviceSpace> & stencil,
                        const TypedFieldBase<T, DefaultDeviceSpace> & x,
                        TypedFieldBase<T, DefaultDeviceSpace> & y, T alpha,
                        bool increment) const {
            DeviceGuard device_guard{x.get_device_id()};
            const auto g{this->validate(stencil, x, y, true)};
            const Index_t node0{this->first_interior(x, g.node_stride)};
            const Index_t sten0{
                this->first_interior(stencil, g.stencil_stride)};
            variable_stencil::apply_gpu<T>(
                stencil.view().data() + sten0, x.view().data() + node0,
                y.view().data() + node0, this->nb_components, NB_HALF_OFFSETS,
                g, alpha, increment);
        }
#endif

        Index_t nb_components;
    };

    using VariableStencilOperator2D = VariableStencilOperator<2>;
    using VariableStencilOperator3D = VariableStencilOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_VARIABLE_STENCIL_HH_
