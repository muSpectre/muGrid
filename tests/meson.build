#
# C++ unit tests (through Boost's framework)
#

boost_test = dependency('boost', modules: ['unit_test_framework'],
                        required: false)

if boost_test.found()
    mugrid_test_sources = [
        'main_test_suite.cc',
        'header_test_ccoord_operations.cc',
        'header_test_eigen_tools.cc',
        'header_test_ref_array.cc',
        'header_test_t4_map.cc',
        'header_test_tensor_algebra.cc',
        'test_ccoord_operations.cc',
        'test_discrete_gradient_operator.cc',
        'test_field.cc',
        'test_field_collection.cc',
        'test_field_map.cc',
        'test_goodies.cc',
        'test_mapped_fields.cc',
        'test_mapped_state_fields.cc',
        'test_options_dictionary.cc',
        'test_raw_memory_operations.cc',
        'test_state_field_maps.cc',
        'test_state_fields.cc',
        'test_units.cc'
    ]

    if mugrid_with_netcdf
        mugrid_test_sources += [
            'io_test_file_io_base.cc',
            'io_test_file_io_netcdf.cc'
        ]
    endif

    main_test_suite = executable('mugrid_main_test_suite',
        mugrid_test_sources,
        dependencies: [mugrid, boost_test])

    test('mugrid_main_test_suite', main_test_suite)
endif

#
# Test environment
#

envdata = environment()
python_paths = [
    join_paths(meson.current_build_dir(), '../language_bindings/python'),
    join_paths(meson.current_source_dir(), '../language_bindings/python')
]
envdata.append('PYTHONPATH', python_paths)
envdata.append('TESTS_BUILDDIR', meson.current_build_dir())

#
# Python unit tests
#

test('mugrid_python_binding_tests',
    python,
    args: [files('python_binding_tests.py'), '-v', '-f'],
    env: envdata,
    depends: [mugrid_python],
    timeout: test_timeout)

#
# If we have MPI, we also want to run the MPI tests
#

if mpi.found()
    if boost_test.found()
        mugrid_mpi_test_sources = [
            'main_test_suite.cc',
            'mpi_test_communicator.cc',
            'mpi_test_field_map.cc'
        ]

        if mugrid_with_netcdf
            mugrid_mpi_test_sources += [
                'io_mpi_test_file_io_netcdf.cc'
            ]
        endif

        mpi_main_test_suite = executable('mpi_mugrid_main_test_suite',
            mugrid_mpi_test_sources,
            dependencies: [mugrid, boost_test])
    endif

    mpiexec = find_program('mpiexec', required: false)

    if mpiexec.found()
        foreach n : mpi_processes
            if boost_test.found()
                test('mpi_mugrid_main_test_suite-np_' + n,
                    mpiexec,
                    args: ['-n', n, '--oversubscribe', mpi_main_test_suite],
                    env: envdata,
                    is_parallel: false)  # Don't run any other test in parallel
            endif

            test('mpi_mugrid_python_binding_tests',
                mpiexec,
                args: ['-n', n, '--oversubscribe', python.path(),
                        files('python_mpi_binding_test.py'), '-v', '-f'],
                env: envdata,
                depends: [mugrid_python],
                timeout: test_timeout,  # Test can take a long time
                is_parallel: false)  # Don't run any other test in parallel
        endforeach
    endif
endif
