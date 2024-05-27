Getting Started
~~~~~~~~~~~~~~~

Obtaining *µ*\Grid
******************

*µ*\Grid is hosted on a git repository on `GitHub <https://github.com/>`_. To clone it, run

.. code-block:: sh

   $ git clone https://github.com/muSpectre/muGrid.git

or if you prefer identifying yourself using a public ssh-key, run

.. code-block:: bash

   $ git clone git@github.com:muSpectre/muGrid.git

The latter option requires you to have a user account on `GitHub`_.

Building *µ*\Grid
*****************
*µ*\Grid uses `Meson <https://mesonbuild.com/>`_ (0.42.0 or higher) as its build system.

The current (and possibly incomplete list of) dependencies are

- `Meson <https://mesonbuild.com/>`_
- `git <https://git-scm.com/>`_
- `Python3 <https://www.python.org/>`_ including the header files
- `numpy <http://www.numpy.org/>`_

The following dependencies are included as Meson subprojects:

- `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ (2.2.4 or higher)
- `Eigen <http://eigen.tuxfamily.org/>`_ (3.4.0 or higher)

The following dependencies are optional:

- `Boost unit test framework <http://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html>`_
- `Unidata NetCDF <https://unidata.github.io/netcdf4-python/netCDF4/index.html>`_
- `PnetCDF <https://parallel-netcdf.github.io/>`_

Recommended:

- `Sphinx <http://www.sphinx-doc.org>`_ and `Breathe
  <https://breathe.readthedocs.io>`_ (necessary if you want to build the
  documentation (turned off by default)

*µ*\Grid requires a relatively modern compiler as it makes heavy use of C++17 features.

To compile for *development*, i.e. with debug options turned on, first setup
the build folder:

.. code-block:: sh

   $ meson setup meson-build-debug

To compile for *production*, i.e. with code optimizations turned on, setup the
build folder while specifying the `release` build type.

.. code-block:: sh

   $ meson setup --buildtype release meson-build-release

The compilation is typically handled with `ninja <https://ninja-build.org/>`_.
Navigate to the build folder and run:

.. code-block:: sh

   $ meson compile

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
