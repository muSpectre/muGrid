#
# @file   fft.py
#
# @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
#
# @date   27 Mar 2018
#
# @brief  Wrapper for muSpectre's FFT engines
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

import mpi4py

import _muSpectre

_factories = {'fftw': ('FFTW_2d', 'FFTW_3d', False),
              'fftwmpi': ('FFTWMPI_2d', 'FFTWMPI_3d', True),
              'pfft': ('PFFT_2d', 'PFFT_3d', True),
              'p3dfft': ('P3DFFT_2d', 'P3DFFT_3d', True)}


def FFT(resolutions, fft='fftw', communicator=None):
    """
    Instantiate a muSpectre FFT class.

    Parameters
    ----------
    resolutions: list
        Grid resolutions in the Cartesian directions.
    fft: string
        FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and 'p3dfft'.
        Default is 'fftw'.
    communicator: mpi4py communicator
        mpi4py communicator object passed to parallel FFT engines. Note that
        the default 'fftw' engine does not support parallel execution.


    Returns
    -------
    cell: object
        Return a muSpectre Cell object.
    """
    try:
        factory_name_2d, factory_name_3d, is_parallel = _factories[fft]
    except KeyError:
        raise KeyError("Unknown FFT engine '{}'.".format(fft))
    if len(resolutions) == 2:
        factory_name = factory_name_2d
    elif len(resolutions) == 3:
        factory_name = factory_name_3d
    else:
        raise ValueError('{}-d transforms are not supported'
                         .format(len(resolutions)))
    try:
        factory = _muSpectre.fft.__dict__[factory_name]
    except KeyError:
        raise KeyError("FFT engine '{}' has not been compiled into the "
                       "muSpectre library.".format(fft))
    if is_parallel:
        if communicator is None:
            communicator = mpi4py.MPI.COMM_SELF
        return factory(resolutions, mpi4py.MPI._handleof(communicator))
    else:
        if communicator is not None:
            raise ValueError("FFT engine '{}' does not support parallel "
                             "execution.".format(fft))
        return factory(resolutions)
