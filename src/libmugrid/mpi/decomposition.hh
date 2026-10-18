#ifndef SRC_LIBMUGRID_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_DECOMPOSITION_HH_

#include "core/types.hh"
#include "collection/field_collection_global.hh"

namespace muGrid {
    //! Abstract interface of a domain decomposition with ghost layers:
    //! communicate_ghosts() fills ghost entries from their owners (a linear
    //! map G) and reduce_ghosts() applies its transpose G^T. See
    //! CartesianDecomposition for the concrete operators.
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
        //! This is the adjoint operation of communicate_ghosts, for code that
        //! scatters into ghost buffers. muGrid's operators do not: their
        //! transposes gather from communicated input ghosts and write only
        //! owned points, so their results must not be reduced.
        //! After the operation, ghost buffers are zeroed.
        virtual void reduce_ghosts(const Field & field) const = 0;

        //! accumulate ghost buffer contributions back to the interior domain.
        virtual void reduce_ghosts(const std::string & field_name) const = 0;
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_DECOMPOSITION_HH_
