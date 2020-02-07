/**
 * @file   version.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Feb 2020
 *
 * @brief  File written at compile time containing git state
 *
 * Copyright © 2020 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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


#include <string>
#include <sstream>

#include <libmufft/mufft_common.hh>

namespace muSpectre {
  namespace version {

    /* Presently, version informations is inherited from muFFT. If the
     * repositories are split sometime in the future, this need to be created
     * via CMake. */

    //------------------------------------------------------------------------//
    std::string info() {
      std::stringstream info_str{};
      info_str << "µSpectre version: " << muFFT::version::description()
               << std::endl;
      if (muFFT::version::is_dirty()) {
        info_str << "WARNING: state is dirty, you will not be able to recover "
                    "the state of the µSpectre sources used to compile me from "
                    "this info!"
                 << std::endl;
      }
      return info_str.str();
    }

    //------------------------------------------------------------------------//
    const char * hash() { return muFFT::version::hash(); }

    //------------------------------------------------------------------------//
    const char * description() { return muFFT::version::description(); }

    //------------------------------------------------------------------------//
    bool is_dirty() { return muFFT::version::is_dirty(); }

  }  // namespace version

}  // namespace muSpectre
