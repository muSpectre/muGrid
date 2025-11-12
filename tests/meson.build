#
# C++ unit tests (through Boost's framework)
#
boost_test = dependency('boost', modules : ['unit_test_framework'], required : false)

# Check for PAPI (Performance API) for performance testing
papi = dependency('papi', required : false)
if papi.found()
    message('PAPI found - performance tests will be enabled')
else
    message('PAPI not found - performance tests will be skipped')
endif

if boost_test.found()
    #
    # General tests
    #
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
        'test_units.cc',
    ]
    if mugrid_with_netcdf
        mugrid_test_sources += ['io_test_file_io_base.cc', 'io_test_file_io_netcdf.cc']
    endif

    main_test_suite = executable(
        'mugrid_main_test_suite',
        mugrid_test_sources,
        dependencies : [mugrid, boost_test],
    )

    test('mugrid_main_test_suite', main_test_suite, timeout : test_timeout)

    #
    # Performance tests with PAPI
    #
    if papi.found()
        performance_test_sources = [
            'main_test_suite.cc',
            'test_convolution_performance.cc',
        ]

        performance_test_suite = executable(
            'mugrid_performance_test_suite',
            performance_test_sources,
            dependencies : [mugrid, boost_test, papi],
        )

        test('mugrid_performance_test_suite', performance_test_suite,
             timeout : test_timeout)
    endif

    #
    # Parallel computing specific tests
    #
    mugrid_mpi_test_sources = [
        'main_test_suite.cc',
        'mpi_test_communicator.cc',
        'mpi_test_decomposition.cc',
        'mpi_test_field_map.cc',
    ]
    if mpi.found() and mugrid_with_netcdf  # this test is not compatible with stub MPI
        mugrid_mpi_test_sources += ['io_mpi_test_file_io_netcdf.cc']
    endif

    mpi_main_test_suite = executable(
        'mpi_mugrid_main_test_suite',
        mugrid_mpi_test_sources,
        dependencies : [mugrid, boost_test],
    )

    if mpi.found()
        mpiexec = find_program('mpiexec', required : false)
        if mpiexec.found()
            foreach n : mpi_processes
                test(
                    'mpi_mugrid_main_test_suite-np_' + n,
                    mpiexec,
                    args : ['-n', n, '--oversubscribe', mpi_main_test_suite],
                    is_parallel : false, # Don't run any other test in parallel
                    timeout : test_timeout,
                )
            endforeach
        else
            message('mpiexec not found, skipping MPI tests')
        endif
    else
        test(
            'mpi_mugrid_main_test_suite-with_stub',
            mpi_main_test_suite,
            timeout : test_timeout,
        )
    endif
endif

#
# Python unit tests
#
envdata = environment()
python_paths = [
    join_paths(meson.current_build_dir(), '../language_bindings/python'),
    join_paths(meson.current_source_dir(), '../language_bindings/python'),
]
envdata.append('PYTHONPATH', python_paths)
envdata.append('TESTS_BUILDDIR', meson.current_build_dir())

#
# General tests on pyton bindings
#
test(
    'mugrid_python_binding_tests',
    python,
    args : ['-m', 'pytest', '-v', '-s', join_paths(meson.project_source_root(), 'tests')],
    env : envdata,
    depends : [mugrid_python],
    timeout : test_timeout,
)

#
# Parallel computing specific tests on python bindings
#
if mpi.found()
    mpiexec = find_program('mpiexec', required : false)
    if mpiexec.found()
        foreach n : mpi_processes
            test(
                'mpi_mugrid_python_binding_tests',
                mpiexec,
                args : [
                    '-n', n,
                    '--oversubscribe', python.full_path(),
                    '-m',
                    'pytest',
                    '-v',
                    '-s',
                    join_paths(meson.project_source_root(), 'tests')
                ],
                env : envdata,
                depends : [mugrid_python],
                timeout : test_timeout, # Test can take a long time
                is_parallel : false, # Don't run any other test in parallel
            )
        endforeach
    else
        message('mpiexec not found, skipping MPI tests')
    endif
else
    test(
        'mugrid_python_binding_tests',
        python,
        args : ['-m', 'pytest', '-v', '-s', join_paths(meson.project_source_root(), 'tests')],
        env : envdata,
        depends : [mugrid_python],
        timeout : test_timeout,
    )
endif