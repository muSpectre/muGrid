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

pybind11 = dependency('pybind11', required: true)

mugrid_python = python.extension_module(
    '_muGrid', # Name of the module
    mugrid_python_sources,
    install: true,  # Install it
    dependencies: [mugrid, pybind11]
)

subdir('muGrid')