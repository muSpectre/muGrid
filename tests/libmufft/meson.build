#
# C++ unit tests (through Boost's framework)
#

incdirs = [eigen3_incdir, include_directories('../../src')]

if mu_with_c_tests
    main_test_suite = executable('main_test_suite',
        'main_test_suite.cc',
        'test_fft_utils.cc',
        'test_serial_fft_engines.cc',
        include_directories: incdirs,
        link_with: [mugrid, mufft, muspectre],
        dependencies: [muspectre_dependencies, boost_test])

    test('mufft_main_test_suite',
        main_test_suite)
endif

#
# Python unit tests
#

test('mufft_python_binding_tests',
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
        mpi_main_test_suite = executable('mpi_main_test_suite',
            'mpi_main_test_suite.cc',
            'mpi_test_fft_engine.cc',
            include_directories: incdirs,
            link_with: [mugrid, mufft, muspectre],
            dependencies: [muspectre_dependencies, boost_test])
    endif

    foreach n : mpi_processes
        if mu_with_c_tests
            test('mpi_mufft_main_test_suite-np_' + n,
                 mpiexec,
                 args: ['-n', n, '--oversubscribe', mpi_main_test_suite],
                 is_parallel: false)  # Don't run any other test in parallel
        endif

        test('mpi_mufft_python_binding_tests',
             mpiexec,
             args: ['-n', n, '--oversubscribe', python.path(),
                    files('python_mpi_binding_tests.py'), '-v', '-f'],
               depends: [mugrid_python, mufft_python, muspectre_python],
               env: envdata,
               timeout: test_timeout,  # Test can take a long time
               is_parallel: false)  # Don't run any other test in parallel
    endforeach
endif
