/**
 * @file   core/version.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Version information for muGrid
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_LIBMUGRID_CORE_VERSION_HH_
#define SRC_LIBMUGRID_CORE_VERSION_HH_

#include <string>

namespace muGrid {
    namespace version {

        /**
         * @brief Returns a formatted text that can be printed to stdout or to
         * output files.
         *
         * This function generates a string that contains the git commit hash
         * and repository url used to compile µGrid. It also indicates whether
         * the current state was dirty or not.
         *
         * @return A formatted string containing the git commit hash, repository
         * url and the state of the repository.
         */
        std::string info();

        /**
         * @brief Returns the git commit hash.
         *
         * This function retrieves the git commit hash used to compile µGrid.
         *
         * @return A constant character pointer representing the git commit
         * hash.
         */
        const char * hash();

        /**
         * @brief Returns the repository description.
         *
         * This function retrieves the repository description used to compile
         * µGrid.
         *
         * @return A constant character pointer representing the repository
         * description.
         */
        const char * description();

        /**
         * @brief Checks if the current state was dirty.
         *
         * This function checks if the current state of the repository used to
         * compile µGrid was dirty or not.
         *
         * @return A boolean value indicating if the state was dirty (true) or
         * not (false).
         */
        bool is_dirty();

    }  // namespace version
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CORE_VERSION_HH_
