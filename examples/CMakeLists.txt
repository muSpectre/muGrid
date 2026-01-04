# Example Python scripts
set(MUGRID_EXAMPLES
    components.py
    fft_roundtrip.py
    field_collection.py
    fourier_derivative.py
    gradient.py
    homogenization.py
    io.py
    poisson.py
    sub_points.py
)

# Use the same Python interpreter as tests (set in tests/CMakeLists.txt)
# Fall back to Python_EXECUTABLE if not set
if(NOT MUGRID_PYTHON_EXECUTABLE)
    set(MUGRID_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH
        "Python interpreter for running examples (should have numpy installed)")
endif()

# Check if numpy is available with this Python
execute_process(
    COMMAND ${MUGRID_PYTHON_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE NUMPY_CHECK_RESULT
    OUTPUT_QUIET ERROR_QUIET
)

if(NUMPY_CHECK_RESULT EQUAL 0)
    # Set up Python path for examples
    set(PYTHON_EXAMPLE_ENV
        "PYTHONPATH=${CMAKE_BINARY_DIR}/language_bindings/python:${CMAKE_SOURCE_DIR}/language_bindings/python"
        "TESTS_BUILDDIR=${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Register each example as a test
    foreach(EXAMPLE ${MUGRID_EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
        add_test(
            NAME example_${EXAMPLE_NAME}
            COMMAND ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}
        )
        set_tests_properties(example_${EXAMPLE_NAME} PROPERTIES
            TIMEOUT 300
            ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
        )
    endforeach()

    # Additional 3D tests for poisson and homogenization examples
    # These ensure both 2D and 3D code paths are tested
    add_test(
        NAME example_poisson_3d
        COMMAND ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py -n 8,8,8 -q
    )
    set_tests_properties(example_poisson_3d PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
    )

    add_test(
        NAME example_homogenization_3d
        COMMAND ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/homogenization.py -n 6,6,6 -q
    )
    set_tests_properties(example_homogenization_3d PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
    )

    # MPI parallel tests for poisson and homogenization examples
    # These test that the examples run correctly with multiple MPI ranks
    if(MUGRID_ENABLE_MPI)
        find_program(MPIEXEC_EXECUTABLE mpiexec)
        if(MPIEXEC_EXECUTABLE)
            # Test poisson example with MPI (2D and 3D)
            foreach(NP 2 4)
                add_test(
                    NAME example_poisson_mpi-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe
                        ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py -n 16,16 -q --maxiter 10
                )
                set_tests_properties(example_poisson_mpi-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                    ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
                )

                add_test(
                    NAME example_poisson_3d_mpi-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe
                        ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py -n 8,8,8 -q --maxiter 10
                )
                set_tests_properties(example_poisson_3d_mpi-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                    ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
                )
            endforeach()

            # Test homogenization example with MPI (2D and 3D)
            foreach(NP 2 4)
                add_test(
                    NAME example_homogenization_mpi-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe
                        ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/homogenization.py -n 12,12 -q --maxiter 50
                )
                set_tests_properties(example_homogenization_mpi-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                    ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
                )

                add_test(
                    NAME example_homogenization_3d_mpi-np_${NP}
                    COMMAND ${MPIEXEC_EXECUTABLE} -n ${NP} --oversubscribe
                        ${MUGRID_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/homogenization.py -n 6,6,6 -q --maxiter 50
                )
                set_tests_properties(example_homogenization_3d_mpi-np_${NP} PROPERTIES
                    TIMEOUT 300
                    PROCESSORS ${NP}
                    ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
                )
            endforeach()
        else()
            message(WARNING "mpiexec not found, skipping MPI example tests")
        endif()
    endif()
else()
    message(STATUS "numpy not available in ${MUGRID_PYTHON_EXECUTABLE}, skipping Python examples")
endif()
