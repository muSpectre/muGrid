/**
 * @file   core/environment.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Oct 2026
 *
 * @brief  Reading muGrid's boolean environment switches
 *
 * Copyright © 2026 Lars Pastewka
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
 */

#include "core/environment.hh"

#include "core/exception.hh"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace muGrid {

    int environment_flag(const char * name) {
        const char * env{std::getenv(name)};
        if (env == nullptr || env[0] == '\0') {
            return -1;
        }
        std::string value{env};
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        for (const char * on : {"1", "y", "yes", "t", "true", "on"}) {
            if (value == on) {
                return 1;
            }
        }
        for (const char * off : {"0", "n", "no", "f", "false", "off"}) {
            if (value == off) {
                return 0;
            }
        }
        throw RuntimeError(std::string(name) + "='" + env +
                           "' is not a boolean; use 1/0, yes/no, true/false "
                           "or on/off");
    }

}  // namespace muGrid
