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
    CartesianDecomposition(const Communicator & comm,
                           const DynCcoord_t & nb_domain_grid_pts,
                           const DynCcoord_t & nb_subdivisions,
                           const DynCcoord_t & nb_ghost_left,
                           const DynCcoord_t & nb_ghost_right,
                           const SubPtMap_t & nb_sub_pts = {});

    CartesianDecomposition() = delete;

    virtual ~CartesianDecomposition() {}

    //! fill the ghost buffers with the values from the neighboring processes.
    void communicate_ghosts(std::string field_name) const;

    //! get the field collection
    GlobalFieldCollection & get_collection() const;

    //! get the number of subdivisions
    const DynCcoord_t get_nb_subdivisions() const;

    //! get the number of grid points of the whole domain
    const DynCcoord_t get_nb_domain_grid_pts() const;

    //! get the number of grid points per subdomain
    const DynCcoord_t get_nb_subdomain_grid_pts() const;

    //! get the subdomain locations
    const DynCcoord_t get_subdomain_locations() const;

   protected:
    std::unique_ptr<GlobalFieldCollection> collection;
    DynCcoord_t nb_ghosts_left;
    DynCcoord_t nb_ghosts_right;
    CartesianCommunicator comm;
  };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_DECOMPOSITION_HH_
