# Python bindings source files
set(MUGRID_PYTHON_SOURCES
    bind_py_module.cc
    bind_py_common_mugrid.cc
    bind_py_communicator.cc
    bind_py_decomposition.cc
    bind_py_convolution_operator.cc
    bind_py_field.cc
    bind_py_state_field.cc
    bind_py_field_collection.cc
    bind_py_file_io.cc
    bind_py_fft.cc
)

# Create Python module
pybind11_add_module(_muGrid MODULE ${MUGRID_PYTHON_SOURCES})

# Link against muGrid library
target_link_libraries(_muGrid PRIVATE muGrid)

# Include directories for bindings
target_include_directories(_muGrid PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/libmugrid
    ${CMAKE_SOURCE_DIR}/src
    ${dlpack_SOURCE_DIR}/include
)

# NetCDF support
if(MUGRID_ENABLE_NETCDF)
    target_compile_definitions(_muGrid PRIVATE WITH_NETCDF_IO)
    target_include_directories(_muGrid PRIVATE ${MUGRID_NETCDF_INCLUDE_DIRS})
endif()

# Pass device architecture information to Python bindings
if(MUGRID_ENABLE_CUDA AND CMAKE_CUDA_ARCHITECTURES)
    # Convert list to semicolon-separated string for display
    string(REPLACE ";" "," MUGRID_DEVICE_ARCH_STR "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_definitions(_muGrid PRIVATE MUGRID_DEVICE_ARCH=${MUGRID_DEVICE_ARCH_STR})
elseif(MUGRID_ENABLE_HIP AND CMAKE_HIP_ARCHITECTURES)
    # Convert list to semicolon-separated string for display
    string(REPLACE ";" "," MUGRID_DEVICE_ARCH_STR "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_definitions(_muGrid PRIVATE MUGRID_DEVICE_ARCH=${MUGRID_DEVICE_ARCH_STR})
endif()

# Install the Python module
# Use relative path for scikit-build-core wheel compatibility
install(TARGETS _muGrid
    LIBRARY DESTINATION muGrid
    COMPONENT python
)

# Install pure Python files
install(FILES
    muGrid/__init__.py
    muGrid/Field.py
    muGrid/Parallel.py
    muGrid/Solvers.py
    muGrid/Timer.py
    muGrid/Wrappers.py
    DESTINATION muGrid
    COMPONENT python
)
