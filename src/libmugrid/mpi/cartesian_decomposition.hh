#ifndef SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_

#include "core/coordinates.hh"
#include "collection/field_collection_global.hh"
#include "mpi/communicator.hh"
#include "mpi/cartesian_communicator.hh"
#include "mpi/decomposition.hh"

namespace muGrid {
    /**
     * Block decomposition of a periodic d-dimensional grid of
     * N_0 x ... x N_{d-1} points over P = P_0 x ... x P_{d-1} ranks arranged
     * on a periodic Cartesian topology, with ghost layers of width
     * gL_a (left) and gR_a (right) along each axis a.
     *
     * When the subdomains are derived from the subdivisions, a rank with
     * Cartesian coordinate c_a owns n_a = floor(N_a / P_a) + 1 points if
     * c_a < N_a mod P_a and floor(N_a / P_a) otherwise, starting at global
     * index s_a = c_a floor(N_a / P_a) + min(c_a, N_a mod P_a); subdomains
     * may instead be passed in explicitly. Its local array has n_a + gL_a + gR_a points along a; local index i
     * corresponds to global index (s_a - gL_a + i) mod N_a, and the interior
     * is gL_a <= i < gL_a + n_a.
     *
     * communicate_ghosts() applies the linear operator G = G_{d-1} ... G_0,
     * where G_a overwrites every ghost slice along axis a with the entry of
     * the same global index (mod N_a) on the rank that holds it. Each G_a
     * transfers whole slices normal to a, including the ghost layers of the
     * other axes, so edge and corner ghosts are filled by the composition.
     * reduce_ghosts() applies the exact transpose
     * G^T = G_0^T ... G_{d-1}^T: every ghost entry is added to the entry it
     * aliases and then set to zero. With the inner product summed over all
     * local entries (ghosts included) of all ranks,
     * <G x, y> = <x, G^T y> holds exactly.
     *
     * A ghost layer may be wider than a neighbour's interior; the missing
     * slices are then relayed over several sendrecv steps through the
     * intermediate ranks (see recv_right_sequence).
     *
     * The ghost exchange requires column-major pixel storage (first axis
     * fastest); the per-pixel dof layout may be AoS or SoA.
     */
    class CartesianDecomposition : public Decomposition {
       public:
        using Parent_t = Decomposition;
        using SubPtMap_t = FieldCollection::SubPtMap_t;

        /**
         * Constructor with deferred initialization (call initialise() before
         * use)
         * @param comm communicator
         * @param spatial_dimension spatial dimension of the problem
         * @param nb_sub_pts number of sub-points per pixel
         * @param device where to allocate field memory
         */
        CartesianDecomposition(const Communicator & comm,
                               Dim_t spatial_dimension,
                               const SubPtMap_t & nb_sub_pts = {},
                               Device device = Device::cpu());

        /**
         * Constructor with immediate initialization; subdomains are
         * determined from the subdivisions as described for the class
         * @param comm communicator
         * @param nb_domain_grid_pts number of grid points in the whole domain
         * @param nb_subdivisions number of subdivisions (ranks) in each
         *        direction; their product must equal the communicator size
         * @param nb_ghosts_left number of ghost cells on the left side
         * @param nb_ghosts_right number of ghost cells on the right side
         * @param nb_sub_pts number of sub-points per pixel
         * @param device where to allocate field memory
         */
        CartesianDecomposition(const Communicator & comm,
                               const DynGridIndex & nb_domain_grid_pts,
                               const DynGridIndex & nb_subdivisions,
                               const DynGridIndex & nb_ghosts_left,
                               const DynGridIndex & nb_ghosts_right,
                               const SubPtMap_t & nb_sub_pts = {},
                               Device device = Device::cpu());

        CartesianDecomposition() = delete;

        ~CartesianDecomposition() override = default;

        //! initialise with known subdomains (e.g. as dictated by an FFT
        //! library). Locations and sizes exclude ghosts. An empty
        //! `subdomain_strides` selects column-major pixel strides (first
        //! axis fastest). Collective: negotiates the ghost
        //! exchange schedule with the neighbours.
        void initialise(const DynGridIndex & nb_domain_grid_pts,
                        const DynGridIndex & nb_subdivisions,
                        const DynGridIndex & nb_subdomain_grid_pts_without_ghosts,
                        const DynGridIndex & subdomain_locations_without_ghosts,
                        const DynGridIndex & nb_ghosts_left,
                        const DynGridIndex & nb_ghosts_right,
                        const DynGridIndex & subdomain_strides = DynGridIndex{});

        //! initialise and determine subdomains from subdivisions (balanced
        //! block distribution, see class documentation). Collective.
        void initialise(const DynGridIndex & nb_domain_grid_pts,
                        const DynGridIndex & nb_subdivisions,
                        const DynGridIndex & nb_ghosts_left,
                        const DynGridIndex & nb_ghosts_right);

        //! fill the ghost buffers with the values from the neighboring
        //! processes (periodic), i.e. apply G (see class documentation).
        //! Collective over the communicator.
        void communicate_ghosts(const Field & field) const override;

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        void communicate_ghosts(const std::string & field_name) const override;

        //! accumulate ghost buffer contributions back to the interior domain,
        //! i.e. apply G^T (see class documentation). This is the adjoint
        //! operation of communicate_ghosts, for code that scatters into
        //! ghost buffers (muGrid's operator transposes do not; see
        //! decomposition.hh).
        //! Each ghost slice is added to the interior slice it aliases, on
        //! whichever rank owns it; halos wider than a neighbour's interior
        //! are relayed through the intermediate ranks. After the operation,
        //! ghost buffers are zeroed. Collective over the communicator.
        void reduce_ghosts(const Field & field) const override;

        //! accumulate ghost buffer contributions back to the interior domain.
        void reduce_ghosts(const std::string & field_name) const override;

        //! get the field collection
        GlobalFieldCollection & get_collection();

        //! get the field collection
        const GlobalFieldCollection & get_collection() const;

        //! get the spatial dimension
        virtual Dim_t get_spatial_dim() const;

        //! get the number of subdivisions
        const DynGridIndex & get_nb_subdivisions() const;

        //! get the number of grid points of the whole domain
        virtual const DynGridIndex & get_nb_domain_grid_pts() const;

        //! get the number of grid points of this subdomain, ghosts included
        const DynGridIndex & get_nb_subdomain_grid_pts_with_ghosts() const;

        //! get the number of grid points of this subdomain, ghosts excluded
        DynGridIndex get_nb_subdomain_grid_pts_without_ghosts() const;

        //! get the global index of this subdomain's first point, counting
        //! the left ghosts (may be negative)
        const DynGridIndex & get_subdomain_locations_with_ghosts() const;

        //! get the global index of this subdomain's first interior point
        DynGridIndex get_subdomain_locations_without_ghosts() const;

        //! get the number of ghost cells on the left side
        const DynGridIndex & get_nb_ghosts_left() const {
            return this->collection.get_nb_ghosts_left();
        }

        //! get the number of ghost cells on the right side
        const DynGridIndex & get_nb_ghosts_right() const {
            return this->collection.get_nb_ghosts_right();
        }

        //! check if fields in this decomposition are on device (GPU) memory
        bool is_on_device() const {
            return this->collection.is_on_device();
        }

        //! get the device of the field collection
        Device get_device() const {
            return this->collection.get_device();
        }

       protected:
        Communicator comm;
        std::unique_ptr<CartesianCommunicator> cart_comm;
        GlobalFieldCollection collection;
        //! Ghost exchange schedule, per direction and per communication
        //! step, in units of slices normal to the direction. Filling a halo
        //! wider than a neighbour's interior takes several steps in which
        //! ranks relay slices they received in the previous step; the
        //! schedule is negotiated collectively in initialise() (so it is
        //! identical on all ranks) and replayed by communicate_ghosts()
        //! forwards and by reduce_ghosts() backwards.
        //!
        //! Along axis a with interior n and ghosts gL, gR, step 0 sends
        //! min(n, gL) slices right and min(n, gR) slices left; step k > 0
        //! sends min(gL - (slices sent right so far), slices received from
        //! the left in step k-1) right, and symmetrically left, until every
        //! rank's ghosts are full. Entries are indexed [direction][step].
        //!
        //! Slices received from the right neighbour into the right ghost
        std::vector<std::vector<Index_t>> recv_right_sequence;
        //! Slices received from the left neighbour into the left ghost
        std::vector<std::vector<Index_t>> recv_left_sequence;
        //! Slices sent to the right neighbour (its left ghost)
        std::vector<std::vector<Index_t>> send_right_sequence;
        //! Slices sent to the left neighbour (its right ghost)
        std::vector<std::vector<Index_t>> send_left_sequence;
        //! Number of steps per direction
        std::vector<Index_t> nb_sendrecv_steps;

        //! Throw unless `n` has the spatial dimension of the collection
        void check_dimension(const DynGridIndex & n,
                             const std::string & name) const;

        //! Ghost communication combines this decomposition's grid extents
        //! and ghost counts with the strides of the field it is handed, so
        //! it is only meaningful for fields of this decomposition's own
        //! collection. Fields of any other collection (e.g. Fourier-space
        //! fields of an FFT engine, which live on a half-complex grid
        //! without ghost regions) would have interior data scrambled;
        //! reject them up front.
        void check_field_is_of_this_collection(
            const Field & field, const std::string & operation) const;

        //! Throw unless the field's pixels are column-major (first axis
        //! fastest, possibly padded), the layout the ghost exchange slices.
        void check_ghost_layout(const Field & field,
                                const std::string & operation) const;
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
