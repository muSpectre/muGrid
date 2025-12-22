Getting Started
~~~~~~~~~~~~~~~

Python quick start
******************

To install µGrid's Python bindings, run

.. code-block:: sh

    $ pip install muGrid

Note that on most platforms this will install a binary wheel that was
compiled with a minimal configuration. To compile for your specific platform
use

.. code-block:: sh

    $ pip install -v --force-reinstall --no-cache --no-binary muGrid muGrid

which will compile the code. µGrid will autodetect
`MPI <https://www.mpi-forum.org/>`_.
For I/O, it will try to use
`Unidata NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_
for serial builds and
`PnetCDF <https://parallel-netcdf.github.io/>`_ for MPI-parallel builds.
Monitor output to see which of these options were automatically detected.
You should see something like::

    --------------------
    muGrid configuration
      Version        : 0.95.0-328-gc99adf21
      Eigen          : 5.0.1 (fetched)
      CUDA           : OFF
      HIP            : OFF
      MPI            : ON - MPI 3.1
      Parallel NetCDF: ON - PnetCDF 1.13.0
      Python bindings: ON - Python 3.14.2
      pybind11       : 2.13.6 (fetched)
      Tests          : ON
      Examples       : ON
    --------------------

Obtaining *µ*\Grid's source code
********************************

*µ*\Grid is hosted on a git repository on `GitHub <https://github.com/>`_. To clone it, run

.. code-block:: sh

   $ git clone https://github.com/muSpectre/muGrid.git

or if you prefer identifying yourself using a public ssh-key, run

.. code-block:: bash

   $ git clone git@github.com:muSpectre/muGrid.git

The latter option requires you to have a user account on `GitHub`_.

Building *µ*\Grid
*****************

*µ*\Grid uses `CMake <https://cmake.org/>`_ (3.18 or higher) as its build system.

The current (and possibly incomplete list of) dependencies are:

- `CMake <https://cmake.org/>`_ (3.18 or higher)
- `git <https://git-scm.com/>`_
- `Python3 <https://www.python.org/>`_ (3.10 or higher) including the header files
- `numpy <http://www.numpy.org/>`_

The following dependencies are fetched automatically if not found:

- `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ (2.11 or higher)
- `Eigen <http://eigen.tuxfamily.org/>`_ (5.0.1 or higher)
- `DLPack <https://github.com/dmlc/dlpack>`_ (for GPU array exchange)

The following dependencies are optional:

- `Boost unit test framework <http://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html>`_
- `Unidata NetCDF <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_ (for serial I/O)
- `PnetCDF <https://parallel-netcdf.github.io/>`_ (for parallel I/O with MPI)
- `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ (for NVIDIA GPU support)
- `ROCm/HIP <https://rocm.docs.amd.com/>`_ (for AMD GPU support)

Recommended:

- `Sphinx <http://www.sphinx-doc.org>`_ and `Breathe
  <https://breathe.readthedocs.io>`_ (necessary if you want to build the
  documentation (turned off by default)

*µ*\Grid requires a relatively modern compiler as it makes heavy use of C++20 features.

To compile for *development*, i.e. with debug options turned on, first setup
the build folder:

.. code-block:: sh

   $ cmake -DCMAKE_BUILD_TYPE=Debug -B build-debug
   $ cmake --build build-debug

To compile for *production*, i.e. with code optimizations turned on, setup the
build folder while specifying the `Release` build type:

.. code-block:: sh

   $ cmake -DCMAKE_BUILD_TYPE=Release -B build-release
   $ cmake --build build-release

You can also use `ninja <https://ninja-build.org/>`_ as the build backend for
faster compilation:

.. code-block:: sh

   $ cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -B build-release
   $ cmake --build build-release

Enabling and disabling features
*******************************

By default, CMake autodetects features for you. The build system will print a
configuration summary showing what features are enabled:

.. code-block:: text

   --------------------
   muGrid configuration
     Version        : v0.96.0
     Eigen          : 5.0.1
     CUDA           : OFF
     HIP            : OFF
     MPI            : ON - MPI 3.1
     Parallel NetCDF: ON - PnetCDF 1.12.3
     Python bindings: ON - Python 3.11
     pybind11       : 2.13.6
     Tests          : ON
     Examples       : ON
   --------------------

You can manually enable or disable specific features using CMake options. Here
are the available options:

**Core features:**

.. code-block:: sh

   # Disable MPI (build serial version)
   $ cmake -DMUGRID_ENABLE_MPI=OFF ..

   # Disable NetCDF I/O
   $ cmake -DMUGRID_ENABLE_NETCDF=OFF ..

   # Disable Python bindings
   $ cmake -DMUGRID_ENABLE_PYTHON=OFF ..

   # Disable tests
   $ cmake -DMUGRID_ENABLE_TESTS=OFF ..

   # Disable examples
   $ cmake -DMUGRID_ENABLE_EXAMPLES=OFF ..

**GPU support:**

GPU support is disabled by default and must be explicitly enabled. See the
:doc:`GPU` documentation for detailed usage information.

.. code-block:: sh

   # Enable NVIDIA CUDA support
   $ cmake -DMUGRID_ENABLE_CUDA=ON ..

   # Enable AMD ROCm/HIP support
   $ cmake -DMUGRID_ENABLE_HIP=ON ..

   # Specify CUDA target architectures (70=V100, 80=A100, 90=H100)
   $ cmake -DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80;90" ..

   # Specify HIP target architectures (gfx906=MI50, gfx90a=MI200, gfx942=MI300)
   $ cmake -DMUGRID_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx906;gfx90a" ..

**Combining options:**

You can combine multiple options in a single command:

.. code-block:: sh

   # Build with CUDA but without MPI
   $ cmake -DMUGRID_ENABLE_CUDA=ON -DMUGRID_ENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release ..

   # Build minimal version (no MPI, no NetCDF, no tests)
   $ cmake -DMUGRID_ENABLE_MPI=OFF -DMUGRID_ENABLE_NETCDF=OFF -DMUGRID_ENABLE_TESTS=OFF ..

Getting help and reporting bugs
*******************************

*µ*\Grid is under active development and the documentation
may be spotty. If you run into trouble,
please contact us by opening an `issue
<https://github.com/muSpectre/muGrid/issues>`_ and someone will answer as
soon as possible. You can also check the API :ref:`reference`.

Contribute
**********

We welcome contributions both for new features and bug fixes. New features must
be documented and have unit tests. Please submit merge requests for review.
