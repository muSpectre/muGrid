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

examples = [
    'components.py',
    'field_collection.py',
    'io.py',
    'sub_points.py'
]

foreach example : examples
    test('examples',
         python,
         args : [files(example)],
         env : envdata,
         depends : [mugrid_python],
         timeout : test_timeout)
endforeach
