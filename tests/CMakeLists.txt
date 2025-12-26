# Find Boost.Test
find_package(Boost COMPONENTS unit_test_framework QUIET)

# Find PAPI for performance tests (optional)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PAPI papi QUIET)
endif()
if(PAPI_FOUND)
    message(STATUS "PAPI found - performance tests will be enabled")
else()
    message(STATUS "PAPI not found - performance tests will be skipped")
endif()

if(Boost_FOUND)
    # Serial test sources
    set(MUGRID_TEST_SOURCES
        main_test_suite.cc
        header_test_ccoord_operations.cc
        header_test_ref_array.cc
        header_test_t4_map.cc
        header_test_tensor_algebra.cc
        test_ccoord_operations.cc
        test_convolution_operator.cc
        test_exception.cc
        test_field.cc
        test_field_gpu.cc
        test_field_collection.cc
        test_field_map.cc
        test_goodies.cc
        test_mapped_fields.cc
        test_mapped_state_fields.cc
        test_state_field_maps.cc
        test_state_fields.cc
        test_units.cc
        test_fft_engine.cc
    )

    # Add NetCDF tests if available
    if(MUGRID_ENABLE_NETCDF)
        list(APPEND MUGRID_TEST_SOURCES
            io_test_file_io_base.cc
            io_test_file_io_netcdf.cc
        )
    endif()

    # Main test suite executable
    add_executable(mugrid_main_test_suite ${MUGRID_TEST_SOURCES})
    target_link_libraries(mugrid_main_test_suite PRIVATE
        muGrid
        Boost::unit_test_framework
    )
    if(MUGRID_ENABLE_NETCDF)
        target_link_libraries(mugrid_main_test_suite PRIVATE ${MUGRID_NETCDF_LIBRARIES})
    endif()
    target_include_directories(mugrid_main_test_suite PRIVATE
        ${CMAKE_SOURCE_DIR}/src/libmugrid
        ${CMAKE_SOURCE_DIR}/src
    )

    # Register test
    add_test(NAME mugrid_main_test_suite COMMAND mugrid_main_test_suite)
    set_tests_properties(mugrid_main_test_suite PROPERTIES TIMEOUT 300)

    # MPI test sources
    set(MUGRID_MPI_TEST_SOURCES
        main_test_suite.cc
        mpi_test_communicator.cc
        mpi_test_decomposition.cc
        mpi_test_field_map.cc
    )

    # Add MPI NetCDF test if both MPI and NetCDF are available
    if(MUGRID_ENABLE_MPI AND MUGRID_ENABLE_NETCDF)
        list(APPEND MUGRID_MPI_TEST_SOURCES io_mpi_test_file_io_netcdf.cc)
    endif()

    # MPI test suite executable
    add_executable(mpi_mugrid_main_test_suite ${MUGRID_MPI_TEST_SOURCES})
    target_link_libraries(mpi_mugrid_main_test_suite PRIVATE
        muGrid
        Boost::unit_test_framework
    )
    if(MUGRID_ENABLE_NETCDF)
        target_link_libraries(mpi_mugrid_main_test_suite PRIVATE ${MUGRID_NETCDF_LIBRARIES})
    endif()
    target_include_directories(mpi_mugrid_main_test_suite PRIVATE
        ${CMAKE_SOURCE_DIR}/src/libmugrid
        ${CMAKE_SOURCE_DIR}/src
    )

    # Register MPI tests
    if(MUGRID_ENABLE_MPI)
        find_program(MPIEXEC_EXECUTABLE mpiexec)
        if(MPIEXEC_EXECUTABLE)
            foreach(NP 1 2 4 8)
                add_test(
                    NAME mpi_mugrid_main_test_suite-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe $<TARGET_FILE:mpi_mugrid_main_test_suite>
                )
                set_tests_properties(mpi_mugrid_main_test_suite-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                )
            endforeach()
        else()
            message(WARNING "mpiexec not found, skipping MPI tests")
        endif()
    else()
        # Without MPI, run as serial test
        add_test(NAME mpi_mugrid_main_test_suite-with_stub COMMAND mpi_mugrid_main_test_suite)
        set_tests_properties(mpi_mugrid_main_test_suite-with_stub PROPERTIES TIMEOUT 300)
    endif()
else()
    message(STATUS "Boost.Test not found, skipping C++ unit tests")
endif()

# Python tests
if(MUGRID_ENABLE_PYTHON)
    # Use the Python interpreter that was found during configuration
    # Users should ensure CMake is configured with the correct Python
    # (e.g., by activating venv before running cmake, or by setting -DPython_EXECUTABLE)
    set(MUGRID_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH
        "Python interpreter for running tests (should have numpy and pytest installed)")

    # Check if pytest is available with this Python
    execute_process(
        COMMAND ${MUGRID_PYTHON_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE PYTEST_CHECK_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )

    if(PYTEST_CHECK_RESULT EQUAL 0)
        # Set up Python path for tests
        set(PYTHON_TEST_ENV
            "PYTHONPATH=${CMAKE_BINARY_DIR}/language_bindings/python:${CMAKE_SOURCE_DIR}/language_bindings/python"
            "TESTS_BUILDDIR=${CMAKE_CURRENT_BINARY_DIR}"
        )

        # Serial Python tests
        add_test(
            NAME mugrid_python_binding_tests
            COMMAND ${MUGRID_PYTHON_EXECUTABLE} -m pytest -v -s ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(mugrid_python_binding_tests PROPERTIES
            TIMEOUT 300
            ENVIRONMENT "${PYTHON_TEST_ENV}"
        )

        # MPI Python tests
        if(MUGRID_ENABLE_MPI AND MPIEXEC_EXECUTABLE)
            foreach(NP 1 2 4 8)
                add_test(
                    NAME mpi_mugrid_python_binding_tests-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe
                        ${MUGRID_PYTHON_EXECUTABLE} -m pytest -v -s ${CMAKE_CURRENT_SOURCE_DIR}
                )
                set_tests_properties(mpi_mugrid_python_binding_tests-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                    ENVIRONMENT "${PYTHON_TEST_ENV}"
                )
            endforeach()
        endif()
    else()
        message(STATUS "pytest not available in ${MUGRID_PYTHON_EXECUTABLE}, skipping Python tests")
    endif()
endif()
