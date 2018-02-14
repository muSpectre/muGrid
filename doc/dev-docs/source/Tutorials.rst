Getting Started
~~~~~~~~~~~~~~~

Obtaining **µSpectre**
**********************

**µSpectre** is hosted on a git repository on `c4science`_. To clone it, run

.. code-block:: sh

   $ git clone https://c4science.ch/source/muSpectre.git

or if you prefer identifying yourself using a public ssh-key, run

.. code-block:: bash

   $ git clone ssh://git@c4science.ch/source/muSpectre.git

The latter option requires you to have a user account on c4science (`create <https://c4science.ch/auth/start/>`_).


.. _c4science: https://c4science.ch


Building **µSpectre**
*********************

You can compile **µSpectre** using  `CMake <https://cmake.org/>`_. The current (and possibly incomplete list of) dependencies are

- `Boost unit test framework <http://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html>`_
- `FFTW <http://www.fftw.org>`_
- `Sphinx <http://www.sphinx-doc.org>`_ and `Breathe <https://breathe.readthedocs.io>`_

**µSpectre** requires a relatively modern compiler as it makes heavy use of C++14 features. It has successfully been compiled and tested using the following compilers

- gcc-7.2
- gcc-6.4
- clang-5.0
- clang-4.0

It does *not* compile on Intel's most recent compiler, as it is still lacking some C++14 support. Work-arounds are planned, but will have to wait for someone to pick up the `task <https://c4science.ch/T1852>`_.

To compile, create a build folder and configure the CMake project. If you do this in the folder you cloned in the previous step, it can look for instance like this:

.. code-block:: sh

   $ mkdir build-release
   $ cd build-release
   $ ccmake ..

Then, set the build type to ``Release`` to produce optimised code. µSpectre makes heavy use of expression templates, so optimisation is paramount. (As an example, the performance difference between code compiled in ``Debug`` and ``Release`` is about a factor 40 in simple linear elasticity.)

Finally, compile the library and the tests by running

.. code-block:: sh

   $ make -j <NB-OF-PROCESSES>

.. warning::

   When using the ``-j`` option to compile, be aware that compiling µSpectre uses quite a bit of RAM. If your machine start swapping at compile time, reduce the number of parallel compilations


Running **µSpectre**
********************

The easiest and intended way of using **µSpectre** is through its Python bindings. The following simple example computes the response of a two-dimensional stretched periodic RVE cell. The cell consist of a soft matrix with a circular hard inclusion.

.. literalinclude:: ../../../bin/tutorial_example.py
   :language: python
