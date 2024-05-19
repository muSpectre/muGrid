version_data = configuration_data()
version_data.set('GIT_IS_DIRTY', version_dirty)
version_data.set('GIT_COMMIT_DESCRIBE', version_str)
version_data.set('GIT_HEAD_SHA1', version_hash)
version_file = configure_file(
    input: 'version.cc.skeleton',
    output: 'version.cc',
    configuration: version_data
)

mugrid_sources = [
    'exception.cc',
    'grid_common.cc',
    'ccoord_operations.cc',
    'field_collection.cc',
    'field_collection_global.cc',
    'field_collection_local.cc',
    'file_io_base.cc',
    'field.cc',
    'field_typed.cc',
    'field_map.cc',
    'gradient_operator_default.cc',
    'physics_domain.cc',
    'options_dictionary.cc',
    'raw_memory_operations.cc',
    'state_field.cc',
    'state_field_map.cc',
    'units.cc',
    version_file
]

if mugrid_with_netcdf
    mugrid_sources += [
        'file_io_netcdf.cc',
    ]
endif

if mpi.found()
    mugrid_sources += [
        'communicator.cc'
    ]
endif

libmugrid = shared_library(
    'muGrid',
    mugrid_sources,
    dependencies: mugrid_dependencies,
    install: true  # needed, otherwise it is not copied to wheel
)
