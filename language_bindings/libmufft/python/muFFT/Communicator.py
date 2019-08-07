# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  Main entry point for muFFT Python module

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Lesser Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import _muFFT

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def Communicator(communicator=None):
    """
    Factory function for the communicator class.

    Parameters
    ----------
    communicator: mpi4py or muFFT communicator object
        The bare MPI communicator. (Default: _muFFT.Communicator())
    """
    if communicator is None:
        communicator = _muFFT.Communicator()

    if isinstance(communicator, _muFFT.Communicator):
        return communicator

    if _muFFT.Communicator.has_mpi:
            return _muFFT.Communicator(MPI._handleof(communicator))
    else:
        raise RuntimeError('muFFT was compiled without MPI support.')
