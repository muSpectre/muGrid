#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Parallel.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  muGrid Communicator object

Copyright © 2018 Till Junge

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import _muGrid

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def Communicator(communicator=None):
    """
    Factory function for the communicator class.

    Parameters
    ----------
    communicator: mpi4py or muGrid communicator object
        The bare MPI communicator. (Default: _muGrid.Communicator())
    """
    # If the communicator is None, we return a communicator that contains just
    # the present process.
    if communicator is None:
        communicator = _muGrid.Communicator()

    # If the communicator is already an instance if _muGrid.Communicator, just
    # return that communicator.
    if isinstance(communicator, _muGrid.Communicator):
        return communicator

    # Now we need to do some magic. See if the communicator that was passed
    # conforms with the mpi4py interface, i.e. it has a method 'Get_size'.
    # The present magic enables using either mpi4py or stub implementations
    # of the same interface.
    if hasattr(communicator, 'Get_size'):
        # If the size of the communicator group is 1, just return a
        # communicator that contains just the present process.
        if communicator.Get_size() == 1:
            return _muGrid.Communicator()
        # Otherwise, check if muFFT does actually have MPI support. If yes
        # we assume that the communicator is an mpi4py communicator.
        elif _muGrid.Communicator.has_mpi:
            if not MPI:
                raise RuntimeError('muGrid was compiled with MPI support but '
                                   'mpi4py could not be loaded.')
            return _muGrid.Communicator(MPI._handleof(communicator))
        else:
            raise RuntimeError('muGrid was compiled without MPI support.')
    else:
        raise RuntimeError("The communicator does not have a 'Get_size' "
                           "method. muFFT only supports communicators that "
                           "conform to the mpi4py interface.")
