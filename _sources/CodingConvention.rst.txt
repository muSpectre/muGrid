Coding Convention
~~~~~~~~~~~~~~~~~

This document describes the coding conventions for µGrid. The goal is readable,
maintainable code with consistent style across the project.

Automated Formatting
********************

Code formatting is enforced automatically by tools. Run these before committing:

C++ Code
========

Use ``clang-format`` with the project's ``.clang-format`` configuration:

.. code-block:: bash

    clang-format -i src/libmugrid/**/*.cc src/libmugrid/**/*.hh

Key settings (from ``.clang-format``):

- 4-space indentation
- No tabs
- Pointer alignment: middle (``T * ptr``)
- Namespace contents indented

Python Code
===========

Use Black, isort, and flake8:

.. code-block:: bash

    black .
    isort .
    flake8

Settings are in ``pyproject.toml`` and ``.flake8``:

- Line length: 88 characters (Black default)
- isort profile: black

Pre-commit hooks are configured to run these automatically:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

C++ Conventions
***************

These conventions follow the `Google C++ Style Guide
<https://google.github.io/styleguide/cppguide.html>`_ with the following
project-specific rules.

Language Standard
=================

µGrid uses **C++17**. Use modern C++ features:

- ``auto`` for type inference where it improves readability
- Range-based for loops
- Smart pointers (``std::unique_ptr``, ``std::shared_ptr``) instead of raw pointers
- ``constexpr`` for compile-time constants
- Structured bindings where appropriate

Namespaces
==========

All code lives in the ``muGrid`` namespace:

.. code-block:: c++

    namespace muGrid {

    class MyClass {
        // ...
    };

    }  // namespace muGrid

File Names and Extensions
=========================

- Header files: ``.hh``
- Source files: ``.cc``
- Filenames: lowercase with underscores (e.g., ``fft_engine_base.cc``)

Header Guards
=============

Use ``#ifndef`` guards with the path-based format:

.. code-block:: c++

    #ifndef SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_
    #define SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_

    // ... content ...

    #endif  // SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_

Naming Conventions
==================

Types (classes, structs, enums, type aliases)
    CamelCase: ``FieldCollection``, ``FFTEngine``, ``StorageOrder``

Functions and methods
    lowercase_with_underscores: ``get_field()``, ``compute_fft()``

Variables and parameters
    lowercase_with_underscores: ``nb_grid_pts``, ``field_name``

Class members
    Access via ``this->member_name`` to distinguish from local variables

Constants and enum values
    CamelCase: ``ThreeD``, ``RowMajor``

Template parameters
    Single uppercase letter or CamelCase: ``T``, ``Dim``, ``FieldType``

Include Order
=============

Order includes as follows, with blank lines between groups:

1. µGrid headers
2. Third-party library headers (Eigen, pybind11, etc.)
3. Standard library headers

.. code-block:: c++

    #include "field/field_collection.hh"
    #include "core/types.hh"

    #include <Eigen/Dense>
    #include <pybind11/pybind11.h>

    #include <memory>
    #include <vector>

Documentation
=============

Use Doxygen-style comments for API documentation:

.. code-block:: c++

    /**
     * Brief description of the class.
     *
     * Longer description if needed.
     */
    class MyClass {
     public:
      /**
       * Brief description of method.
       *
       * @param param_name Description of parameter
       * @return Description of return value
       */
      ReturnType method_name(ParamType param_name);

     protected:
      int member_var;  //!< Brief member description
    };

File Headers
============

Every file should start with a license header:

.. code-block:: c++

    /**
     * @file   path/filename.hh
     *
     * @author Your Name <your.email@example.com>
     *
     * @date   DD Mon YYYY
     *
     * @brief  Brief description
     *
     * Copyright © YYYY Author Name
     *
     * µGrid is free software; you can redistribute it and/or
     * modify it under the terms of the GNU Lesser General Public License as
     * published by the Free Software Foundation, either version 3, or (at
     * your option) any later version.
     *
     * ... (rest of license)
     */

Python Conventions
******************

Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ with Black formatting.

Naming Conventions
==================

Classes
    CamelCase: ``FieldCollection``, ``FFTEngine``

Functions and methods
    lowercase_with_underscores: ``get_field()``, ``create_engine()``

Variables
    lowercase_with_underscores: ``nb_grid_pts``, ``field_name``

Constants
    UPPERCASE_WITH_UNDERSCORES: ``MAX_DIM``, ``DEFAULT_TOLERANCE``

Documentation
=============

Use docstrings for all public APIs:

.. code-block:: python

    def create_field(name: str, nb_components: int = 1) -> Field:
        """
        Create a new field in the collection.

        Parameters
        ----------
        name : str
            Unique identifier for the field.
        nb_components : int, optional
            Number of components per grid point. Default is 1.

        Returns
        -------
        Field
            The newly created field.

        Raises
        ------
        ValueError
            If a field with this name already exists.
        """

Testing
*******

C++ Tests
=========

C++ tests use the Boost unit test framework and are located in ``tests/``.
Tests are named ``*_test*.cc``.

Python Tests
============

Python tests use pytest and are located in ``tests/``.
Tests are named ``python_*_tests.py``.

Run tests with:

.. code-block:: bash

    # Build and run C++ tests
    cd builddir
    meson test -v

    # Run Python tests
    pytest tests/

Every new feature should have corresponding tests. Missing tests are considered
bugs.

References
**********

- `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_
- `PEP 8 -- Style Guide for Python Code <https://peps.python.org/pep-0008/>`_
- `Effective Modern C++ by Scott Meyers <http://shop.oreilly.com/product/0636920033707.do>`_
