# Example Python scripts
set(MUGRID_EXAMPLES
    components.py
    field_collection.py
    gradient.py
    io.py
    poisson.py
    sub_points.py
)

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
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}
    )
    set_tests_properties(example_${EXAMPLE_NAME} PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "${PYTHON_EXAMPLE_ENV}"
    )
endforeach()
