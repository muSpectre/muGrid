Code coverage
=============

*µ*\Grid ships with optional code-coverage instrumentation so that untested
code paths can be identified and tracked over time. Coverage is collected for
the C++ library (via ``gcov``/``gcovr``) and for the Python wrapper layer (via
``pytest-cov``).

Building with coverage enabled
------------------------------

Coverage is controlled by the ``MUGRID_ENABLE_COVERAGE`` CMake option. Because
coverage requires an unoptimised build for accurate line mapping, configure with
``CMAKE_BUILD_TYPE=Debug``::

    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DMUGRID_ENABLE_COVERAGE=ON
    cmake --build build

The option works with GCC and Clang. It adds ``--coverage`` together with
``-O0 -g`` and inlining is disabled so that executed lines map back to the
source faithfully. The flags are carried by the ``mugrid::coverage`` interface
target, which the library, the test executables and the Python bindings all
link against, so every component that runs during the test suite is
instrumented.

Collecting and reporting coverage
---------------------------------

First run the test suite to populate the coverage counters, then build one of
the report targets::

    ctest --test-dir build              # C++ and Python tests
    cmake --build build --target coverage

The following convenience targets are available when ``gcovr`` is installed
(``pip install gcovr``):

``coverage-xml``
    Cobertura XML report at ``build/coverage.xml`` (for CI/Codecov upload).

``coverage-html``
    Browsable HTML report at ``build/coverage_html/index.html``.

``coverage-summary``
    Plain-text summary printed to the console.

``coverage``
    Builds all of the above.

The reports cover only ``src/libmugrid``; test code, fetched dependencies and
generated files are excluded.

Python wrapper layer
--------------------

The C++ build above already measures the compiled extension while ``pytest``
runs. To additionally measure the pure-Python wrapper modules
(``muGrid/Field.py``, ``muGrid/Wrappers.py`` and friends), run::

    export PYTHONPATH=$PWD/build/language_bindings/python:$PWD/language_bindings/python
    export TESTS_BUILDDIR=$PWD/build/tests
    pytest tests --cov=muGrid --cov-report=term-missing

Continuous integration
----------------------

The ``Coverage`` GitHub Actions workflow
(``.github/workflows/coverage.yml``) performs an instrumented build on every
push and pull request, runs the full test suite, generates the reports, uploads
them as build artifacts and makes a best-effort upload to Codecov.
