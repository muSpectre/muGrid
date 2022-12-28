#
# C++ unit tests (through Boost's framework)
#

incdirs = [eigen3_incdir, include_directories('../../src')]

if mu_with_c_tests
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

    main_test_suite = executable('main_test_suite',
        mugrid_test_sources,
        include_directories: incdirs,
        link_with: [mugrid, mufft, muspectre],
        dependencies: [muspectre_dependencies, boost_test])

    test('mugrid_main_test_suite', main_test_suite)
endif

#
# Python unit tests
#

test('mugrid_python_binding_tests',
    python,
    args: [files('python_binding_tests.py'), '-v', '-f'],
    depends: [mugrid_python, mufft_python],
    env: envdata,
    timeout: test_timeout)

#
# If we have MPI, we also want to run the MPI tests
#

if mu_with_mpi
    if mu_with_c_tests
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

        mpi_main_test_suite = executable('mpi_main_test_suite',
            mugrid_mpi_test_sources,
            include_directories: incdirs,
            link_with: [mugrid, mufft, muspectre],
            dependencies: [muspectre_dependencies, boost_test])
    endif

    foreach n : mpi_processes
        if mu_with_c_tests
            test('mpi_mugrid_main_test_suite-np_' + n,
                 mpiexec,
                 args: ['-n', n, '--oversubscribe', mpi_main_test_suite],
                 is_parallel: false)  # Don't run any other test in parallel
        endif

        test('mpi_mugrid_python_binding_tests',
             mpiexec,
             args: ['-n', n, '--oversubscribe', python.path(),
                    files('python_mpi_binding_test.py'), '-v', '-f'],
               depends: [mugrid_python, mufft_python, muspectre_python],
               env: envdata,
               timeout: test_timeout,  # Test can take a long time
               is_parallel: false)  # Don't run any other test in parallel
    endforeach
endif
