Getting Started
~~~~~~~~~~~~~~~

Obtaining *µ*\Spectre
*********************

*µ*\Spectre is hosted on a git repository on `gitlab`_. To clone it, run

.. code-block:: sh

   $ git clone https://gitlab.com/muspectre/muspectre.git

or if you prefer identifying yourself using a public ssh-key, run

.. code-block:: bash

   $ git clone git@gitlab.com:muspectre/muspectre.git

The latter option requires you to have a user account on gitlab (`create
<https://gitlab.com/users/sign_in#register-pane>`_).

Note that *µ*\Spectre uses submodules that need to be initialized with

.. code-block:: bash

   $ git submodule update --init

.. _gitlab: https://gitlab.com


Building *µ*\Spectre
********************
You can compile *µ*\Spectre using `Meson <https://mesonbuild.com/>`_
(0.42.0 or higher) or `CMake <https://cmake.org/>`_ (3.5.0 or
higher).

The current (and possibly incomplete list of) dependencies are

- `Meson <https://mesonbuild.com/>`_ (0.42.0 or higher) or `CMake <https://cmake.org/>`_ (3.5.0 or higher)
- `git <https://git-scm.com/>`_
- `Python3 <https://www.python.org/>`_ including the header files
- `numpy <http://www.numpy.org/>`_ and `scipy <https://scipy.org/>`_
- `UVW <https://c4science.ch/source/uvw/>`_ (0.3.1 or higher) for the visualization

The following dependencies are included as submodules:

- `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ (2.2.4 or higher)
- `Eigen <http://eigen.tuxfamily.org/>`_ (3.3.0 or higher)

The following dependencies are optional:

- `Boost unit test framework <http://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html>`_
- `FFTW <http://www.fftw.org>`_
- `NetCDF4 <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_

Recommended:

- `Sphinx <http://www.sphinx-doc.org>`_ and `Breathe
  <https://breathe.readthedocs.io>`_ (necessary if you want to build the
  documentation (turned off by default)
- The CMake curses graphical user interface (``ccmake``).

Possible compilers
 *µ*\Spectre requires a relatively modern compiler as it makes heavy use of C++14
 features. It has successfully been compiled and tested using the following
 compilers under Linux

- gcc-7.2
- gcc-6.4
- gcc-5.4
- clang-6.0
- clang-5.0
- clang-4.0

and using clang-4.0 under MacOS.

Under MacOS 14, it has been successfully compiled using gcc-6.5 shipped with homebrew along with the
CommandLineTools 11.3.
:download:`Here <macosx14_bash_profile>` is an example commented bash_profile that lead to the successful installation.

It does *not* compile on Intel's most recent compiler, as it is still lacking
some C++14 support. Work-arounds are planned, but will have to wait for someone
to pick up the `task <https://gitlab.com/muspectre/muspectre/issues/93>`_.

Instructions for compiling *µ*\Spectre with Meson
*************************************************

To compile for *development*, i.e. with debug options turned on, first setup
the build folder:

.. code-block:: sh

   $ meson setup meson-build-debug

To compile for *production*, i.e. with code optimizations turned on, setup the
build folder while specifying the `release` build type.

.. code-block:: sh

   $ meson setup --buildtype release meson-build-release

The compilation is handled with `ninja`. Navigate to the build folder and run:

.. code-block:: sh

   $ ninja

Instructions for compiling *µ*\Spectre with CMake
*************************************************

To compile, go into the build folder and configure the CMake project. If you do
this in the folder you cloned in the previous step, it can look for instance
like this:

.. code-block:: sh

   $ cd build
   $ ccmake ..

Then, set the build type to ``Release`` to produce optimised code. *µ*\Spectre
makes heavy use of expression templates, so optimisation is paramount. (As an
example, the performance difference between code compiled in ``Debug`` and
``Release`` is about a factor 40 in simple linear elasticity.)

For parallel computation turn ``MUSPECTRE_MPI_PARALLEL`` to ``ON``.

The ``SPLIT_CELL`` option enables the `Laminate Material <./MaterialLaminate.rst>`_

Finally, compile the library and the tests by running

.. code-block:: sh

   $ make -j <NB-OF-PROCESSES>

.. warning::

   When using the ``-j`` option to compile, be aware that compiling *µ*\Spectre
   uses quite a bit of RAM. If your machine start swapping at compile time,
   reduce the number of parallel compilations


Running *µ*\Spectre
*******************

The easiest and intended way of using *µ*\Spectre is through its Python
bindings. The following simple example computes the response of a
two-dimensional stretched periodic RVE cell. The cell consist of a soft matrix
with a circular hard inclusion.

.. literalinclude:: ../../../examples/tutorial_example.py
   :language: python

More examples both python and c++ executables can be found in the ``/examples``
folder.

Getting help
************

*µ*\Spectre is in a very early stage of development and the documentation is
 currently spotty. Also, there is no FAQ page yet. If you run into trouble,
 please contact us by opening an `issue
 <https://gitlab.com/muspectre/muspectre/issues>`_ and someone will answer as
 soon as possible. You can also check the API :ref:`reference`.


Reporting Bugs
**************

If you think you found a bug, you are probably right. Please report it! The
preferred way is for you to create a task on `µSpectre's workboard
<https://gitlab.com/muspectre/muspectre/boards>`_ and assign it to user
``junge``. Include steps to reproduce the bug if possible. Someone will answer
as soon as possible.


Contribute
**********

We welcome contributions both for new features and bug fixes. New features must
be documented and have unit tests. Please submit merge requests for review. More
detailed guidelines for submissions will follow soonᵀᴹ.
