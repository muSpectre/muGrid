#
# We need the Boost unit test framework for C++ tests
#

boost_test = dependency('boost', modules: ['unit_test_framework'],
                        required: false)

mu_with_c_tests = boost_test.found()

#
# Environment for Python tests
#

envdata = environment()
python_paths = [
    # We place the source directories first, because there are no .py files in
    # the build directory, but it does have an empty muGrid, muFFT, muSpectre
    # subdirectory that Python will import as a defunct module.
    join_paths(meson.current_source_dir(), '../language_bindings/libmugrid/python'),
    join_paths(meson.current_source_dir(), '../language_bindings/libmufft/python'),
    join_paths(meson.current_source_dir(), '../language_bindings/python'),
    join_paths(meson.current_build_dir(), '../language_bindings/libmugrid/python'),
    join_paths(meson.current_build_dir(), '../language_bindings/libmufft/python'),
    join_paths(meson.current_build_dir(), '../language_bindings/python')
]
envdata.append('PYTHONPATH', python_paths)
envdata.append('MUSPECTRE_TEST_DIR', meson.current_source_dir())

#
# We need to be able to run MPI programs
#

if mu_with_mpi
    mpiexec = find_program('mpiexec')
endif

subdir('libmugrid')
subdir('libmufft')

incdirs = [eigen3_incdir, include_directories('../src')]

if mu_with_c_tests
    #
    # C++ unit tests (through Boost's framework)
    #

    main_test_suite = executable('main_test_suite',
        'main_test_suite.cc',
        'test_base.cc',
        'test_cell_base.cc',
        'test_cell_data.cc',
        'test_discrete_greens_operator.cc',
        'test_eigen_strain_solver.cc',
        'test_eigen_strain_solver_classes.cc',
        'test_geometry.cc',
        'test_krylov_solvers.cc',
        'test_material_dunant.cc',
        'test_material_dunant_max.cc',
        'test_material_dunant_t.cc',
        'test_material_dunant_tc.cc',
        'test_material_evaluator.cc',
        'test_material_hyper_elasto_plastic1.cc',
        'test_material_hyper_elasto_plastic2.cc',
        'test_material_linear_diffusion.cc',
        'test_material_linear_elastic1.cc',
        'test_material_linear_elastic2.cc',
        'test_material_linear_elastic3.cc',
        'test_material_linear_elastic4.cc',
        'test_material_linear_elastic_damage1.cc',
        'test_material_linear_elastic_damage2.cc',
        'test_material_linear_elastic_generic.cc',
        'test_material_linear_orthotropic.cc',
        'test_material_neo_hookean.cc',
        'test_material_phase_field_fracture.cc',
        'test_material_stochastic_plasticity.cc',
        'test_materials_toolbox.cc',
        'test_material_visco_elastic_damage_ss1.cc',
        'test_material_visco_elastic_damage_ss2.cc',
        'test_material_visco_elastic_ss.cc',
        'test_native_stress_storage.cc',
        'test_projection_finite.cc',
        'test_projection_finite_discrete.cc',
        'test_projection_small.cc',
        'test_solver_fem_newton_cg_class.cc',
        'test_solver_fem_trust_region_newton_cg_class.cc',
        'test_solver_newton_cg.cc',
        'test_solver_newton_cg_class.cc',
        'test_solver_trust_region_newton_cg.cc',
        'test_stiffness_operator.cc',
        'test_s_t_material_linear_elastic_generic1.cc',
        'test_stress_transformation_Kirchhoff_Gradient.cc',
        'test_stress_transformation_Kirchhoff_GreenLagrange.cc',
        'test_stress_transformation_PK2_GreenLagrange.cc',
    #    'split_test_corkpp.cc',
    #    'split_test_intersection_error_induced.cc',
    #    'split_test_laminate_solver.cc',
    #    'split_test_material_laminate.cc',
    #    'split_test_patch_split_cell.cc',
        include_directories: incdirs,
        link_with: [mugrid, mufft, muspectre],
        dependencies: [muspectre_dependencies, boost_test])

    test('muspectre_main_test_suite', main_test_suite)
endif

#
# Python unit tests
#

test('muspectre_python_binding_tests',
    python,
    args: [files('python_binding_tests.py'), '-v', '-f'],
    depends: [mugrid_python, mufft_python, muspectre_python],
    env: envdata,
    timeout: test_timeout)

if mu_with_mpi
    if mu_with_c_tests
        mpi_main_test_suite = executable('mpi_main_test_suite',
            'mpi_main_test_suite.cc',
            'mpi_test_projection_finite.cc',
            'mpi_test_projection_small.cc',
            'mpi_test_solver_newton_cg.cc',
            include_directories: incdirs,
            link_with: [mugrid, mufft, muspectre],
            dependencies: [muspectre_dependencies, boost_test])
    endif

    foreach n : mpi_processes
        if mu_with_c_tests
            test('mpi_muspectre_main_test_suite-np_' + n,
                 mpiexec,
                 args: ['-n', n, '--oversubscribe', mpi_main_test_suite],
                 is_parallel: false)  # Don't run any other test in parallel
        endif

        test('mpi_muspectre_python_binding_tests-np_' + n,
             mpiexec,
             args: ['-n', n, '--oversubscribe', python.path(),
                    files('python_mpi_binding_tests.py'), '-v', '-f'],
               depends: [mugrid_python, mufft_python, muspectre_python],
               env: envdata,
               timeout: test_timeout,  # Test can take a long time
               is_parallel: false)  # Don't run any other test in parallel
    endforeach
endif
