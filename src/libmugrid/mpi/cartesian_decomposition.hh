#ifndef SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_

#include "core/coordinates.hh"
#include "collection/field_collection_global.hh"
#include "mpi/communicator.hh"
#include "mpi/cartesian_communicator.hh"
#include "mpi/decomposition.hh"

namespace muGrid {
    class CartesianDecomposition : public Decomposition {
       public:
        using Parent_t = Decomposition;
        using SubPtMap_t = FieldCollection::SubPtMap_t;

        /*
         * Constructor with deferred initialization
         * @param comm communicator
         * @param spatial_dimension spatial dimension of the problem
         * @param nb_sub_pts number of sub-points per pixel
         * @param device where to allocate field memory
         */
        CartesianDecomposition(const Communicator & comm,
                               Dim_t spatial_dimension,
                               const SubPtMap_t & nb_sub_pts = {},
                               Device device = Device::cpu());

        /*
         * Constructor with immediate initialization
         * @param comm communicator
         * @param nb_domain_grid_pts number of grid points in the whole domain
         * @param nb_subdivisions number of subdivisions in each direction
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

        //! initialise with known subdomains
        void initialise(const DynGridIndex & nb_domain_grid_pts,
                        const DynGridIndex & nb_subdivisions,
                        const DynGridIndex & nb_subdomain_grid_pts_without_ghosts,
                        const DynGridIndex & subdomain_locations_without_ghosts,
                        const DynGridIndex & nb_ghosts_left,
                        const DynGridIndex & nb_ghosts_right,
                        const DynGridIndex & subdomain_strides = DynGridIndex{});

        //! initialise and determine subdomains from subdivisions
        void initialise(const DynGridIndex & nb_domain_grid_pts,
                        const DynGridIndex & nb_subdivisions,
                        const DynGridIndex & nb_ghosts_left,
                        const DynGridIndex & nb_ghosts_right);

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        void communicate_ghosts(const Field & field) const override;

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        void communicate_ghosts(const std::string & field_name) const override;

        //! accumulate ghost buffer contributions back to the interior domain.
        //! This is the adjoint operation of communicate_ghosts and is needed
        //! for transpose operations (e.g., divergence) with periodic BCs.
        //! After the operation, ghost buffers are zeroed.
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

        //! get the number of grid points per subdomain
        const DynGridIndex & get_nb_subdomain_grid_pts_with_ghosts() const;

        //! get the number of grid points per subdomain
        DynGridIndex get_nb_subdomain_grid_pts_without_ghosts() const;

        //! get the subdomain locations
        const DynGridIndex & get_subdomain_locations_with_ghosts() const;

        //! get the subdomain locations
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
        std::vector<std::vector<Index_t>> recv_right_sequence;
        std::vector<std::vector<Index_t>> recv_left_sequence;
        std::vector<Index_t> nb_sendrecv_steps;

        void check_dimension(const DynGridIndex & n,
                             const std::string & name) const;
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
