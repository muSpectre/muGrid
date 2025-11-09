#ifndef SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_

#include "grid_common.hh"
#include "field_collection_global.hh"
#include "communicator.hh"
#include "cartesian_communicator.hh"
#include "decomposition.hh"

namespace muGrid {
    class CartesianDecomposition : public Decomposition {
       public:
        using Parent_t = Decomposition;
        using SubPtMap_t = FieldCollection::SubPtMap_t;

        /*
         * Constructor with deferred initialization
         */
        CartesianDecomposition(const Communicator & comm,
                               Index_t spatial_dimension,
                               const SubPtMap_t & nb_sub_pts = {});

        /*
         * Constructor with immediate initialization
         */
        CartesianDecomposition(const Communicator & comm,
                               const IntCoord_t & nb_domain_grid_pts,
                               const IntCoord_t & nb_subdivisions,
                               const IntCoord_t & nb_ghosts_left,
                               const IntCoord_t & nb_ghosts_right,
                               const SubPtMap_t & nb_sub_pts = {});

        CartesianDecomposition() = delete;

        ~CartesianDecomposition() override = default;

        //! initialise with known subdomains
        void initialise(const IntCoord_t & nb_domain_grid_pts,
                        const IntCoord_t & nb_subdivisions,
                        const IntCoord_t & nb_subdomain_grid_pts_without_ghosts,
                        const IntCoord_t & subdomain_locations_without_ghosts,
                        const IntCoord_t & nb_ghosts_left,
                        const IntCoord_t & nb_ghosts_right,
                        const IntCoord_t & subdomain_strides = IntCoord_t{});

        //! initialise and determine subdomains from subdivisions
        void initialise(const IntCoord_t & nb_domain_grid_pts,
                        const IntCoord_t & nb_subdivisions,
                        const IntCoord_t & nb_ghosts_left,
                        const IntCoord_t & nb_ghosts_right);

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        void communicate_ghosts(const Field & field) const override;

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        void communicate_ghosts(const std::string & field_name) const override;

        //! get the field collection
        GlobalFieldCollection & get_collection();

        //! get the field collection
        const GlobalFieldCollection & get_collection() const;

        //! get the spatial dimension
        virtual Index_t get_spatial_dim() const;

        //! get the number of subdivisions
        const IntCoord_t & get_nb_subdivisions() const;

        //! get the number of grid points of the whole domain
        virtual const IntCoord_t & get_nb_domain_grid_pts() const;

        //! get the number of grid points per subdomain
        const IntCoord_t & get_nb_subdomain_grid_pts_with_ghosts() const;

        //! get the number of grid points per subdomain
        IntCoord_t get_nb_subdomain_grid_pts_without_ghosts() const;

        //! get the subdomain locations
        const IntCoord_t & get_subdomain_locations_with_ghosts() const;

        //! get the subdomain locations
        IntCoord_t get_subdomain_locations_without_ghosts() const;

        //! get the number of ghost cells on the left side
        const IntCoord_t & get_nb_ghosts_left() const {
            return this->collection.get_nb_ghosts_left();
        }

        //! get the number of ghost cells on the right side
        const IntCoord_t & get_nb_ghosts_right() const {
            return this->collection.get_nb_ghosts_right();
        }

       protected:
        Communicator comm;
        std::unique_ptr<CartesianCommunicator> cart_comm;
        GlobalFieldCollection collection;
        std::vector<std::vector<Index_t>> recv_right_sequence;
        std::vector<std::vector<Index_t>> recv_left_sequence;
        std::vector<Index_t> nb_sendrecv_steps;

        void check_dimension(const IntCoord_t & n,
                             const std::string & name) const;
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
