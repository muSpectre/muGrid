mugrid_python_sources = [
    'bind_py_module.cc',
    'bind_py_common_mugrid.cc',
    'bind_py_communicator.cc',
    'bind_py_field.cc',
    'bind_py_state_field.cc',
    'bind_py_field_collection.cc',
    'bind_py_file_io.cc',
    'bind_py_options_dictionary.cc'
]

mugrid_python = python.extension_module(
    '_muGrid', # Name of the module
    mugrid_python_sources,
    install: true,  # Install it
    include_directories: [eigen3_incdir, pybind11_incdir, include_directories('../../../src')],
    link_with: mugrid,
    dependencies: [muspectre_dependencies, python3_dependency]
)

subdir('muGrid')