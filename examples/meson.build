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

test('serial',
     python,
     args: [files('field_collection.py')],
     env: envdata,
     depends: [mugrid_python],
     timeout: test_timeout)
