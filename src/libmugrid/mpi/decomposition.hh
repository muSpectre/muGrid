#ifndef SRC_LIBMUGRID_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_DECOMPOSITION_HH_

#include "core/types.hh"
#include "collection/field_collection_global.hh"

namespace muGrid {
    class Decomposition {
       public:
        Decomposition() = default;

        virtual ~Decomposition() = default;

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        virtual void communicate_ghosts(const Field & field) const = 0;

        //! fill the ghost buffers with the values from the neighboring
        //! processes.
        virtual void
        communicate_ghosts(const std::string & field_name) const = 0;

        //! accumulate ghost buffer contributions back to the interior domain.
        //! This is the adjoint operation of communicate_ghosts and is needed
        //! for transpose operations (e.g., divergence) with periodic BCs.
        //! After the operation, ghost buffers are zeroed.
        virtual void reduce_ghosts(const Field & field) const = 0;

        //! accumulate ghost buffer contributions back to the interior domain.
        virtual void reduce_ghosts(const std::string & field_name) const = 0;
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_DECOMPOSITION_HH_
