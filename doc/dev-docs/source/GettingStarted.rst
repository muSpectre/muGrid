Getting Started
~~~~~~~~~~~~~~~

Obtaining *µ*\Spectre
*********************

*µ*\Spectre is hosted on a git repository on `c4science`_. To clone it, run

.. code-block:: sh

   $ git clone https://c4science.ch/source/muSpectre.git

or if you prefer identifying yourself using a public ssh-key, run

.. code-block:: bash

   $ git clone ssh://git@c4science.ch/source/muSpectre.git

The latter option requires you to have a user account on c4science (`create <https://c4science.ch/auth/start/>`_).


.. _c4science: https://c4science.ch


Building *µ*\Spectre
********************

You can compile *µ*\Spectre using  `CMake <https://cmake.org/>`_. The current (and possibly incomplete list of) dependencies are

- `Boost unit test framework <http://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html>`_
- `FFTW <http://www.fftw.org>`_
- `Sphinx <http://www.sphinx-doc.org>`_ and `Breathe <https://breathe.readthedocs.io>`_
- `git <https://git-scm.com/>`_

*µ*\Spectre requires a relatively modern compiler as it makes heavy use of C++14 features. It has successfully been compiled and tested using the following compilers under Linux

- gcc-7.2
- gcc-6.4
- gcc-5.4
- clang-6.0
- clang-5.0
- clang-4.0

and using clang-4.0 under MacOS.

It does *not* compile on Intel's most recent compiler, as it is still lacking some C++14 support. Work-arounds are planned, but will have to wait for someone to pick up the `task <https://c4science.ch/T1852>`_.

To compile, create a build folder and configure the CMake project. If you do this in the folder you cloned in the previous step, it can look for instance like this:

.. code-block:: sh

   $ mkdir build-release
   $ cd build-release
   $ ccmake ..

Then, set the build type to ``Release`` to produce optimised code. *µ*\Spectre makes heavy use of expression templates, so optimisation is paramount. (As an example, the performance difference between code compiled in ``Debug`` and ``Release`` is about a factor 40 in simple linear elasticity.)

Finally, compile the library and the tests by running

.. code-block:: sh

   $ make -j <NB-OF-PROCESSES>

.. warning::

   When using the ``-j`` option to compile, be aware that compiling *µ*\Spectre uses quite a bit of RAM. If your machine start swapping at compile time, reduce the number of parallel compilations


Running *µ*\Spectre
*******************

The easiest and intended way of using *µ*\Spectre is through its Python bindings. The following simple example computes the response of a two-dimensional stretched periodic RVE cell. The cell consist of a soft matrix with a circular hard inclusion.

.. literalinclude:: ../../../bin/tutorial_example.py
   :language: python

More examples both both python and c++ executables can be found in the ``/bin`` folder.

Getting help
************

*µ*\Spectre is in a very early stage of development and the documentation is currently spotty. Also, there is no FAQ page yet. If you run into trouble, please contact us on the `*µ*\Spectre chat room <https://c4science.ch/Z69>`_ and someone will answer as soon as possible. You can also check the API :ref:`reference`.


Reporting Bugs
**************

If you think you found a bug, you are probably right. Please report it! The preferred way is for you to create a task on `*µ*\Spectre's workboard <https://c4science.ch/project/board/1447/>`_ and assign it to user ``junge``. Include steps to reproduce the bug if possible. Someone will answer as soon as possible.


Contribute
**********

We welcome contributions both for new features and bug fixes. New features must be documented and have unit tests. Please submit contributions for review as `Arcanist revisions <https://secure.phabricator.com/book/phabricator/article/arcanist/>`_. More detailed guidelines for submissions will follow soonᵀᴹ.
