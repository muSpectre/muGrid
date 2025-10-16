#ifndef SRC_LIBMUGRID_DECOMPOSITION_HH_
#define SRC_LIBMUGRID_DECOMPOSITION_HH_

#include "grid_common.hh"
#include "field_collection_global.hh"
#include "communicator.hh"


namespace muGrid {
    class Decomposition {
    public:
        Decomposition() = default;

        virtual ~Decomposition() = default;

        //! fill the ghost buffers with the values from the neighboring processes.
        virtual void communicate_ghosts(const Field &field) const = 0;

        //! fill the ghost buffers with the values from the neighboring processes.
        virtual void communicate_ghosts(const std::string &field_name) const = 0;
    };
} // namespace muGrid

#endif // SRC_LIBMUGRID_DECOMPOSITION_HH_
