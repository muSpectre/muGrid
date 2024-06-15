project(
    'muGrid', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for std::optional
    version : run_command('python3', 'discover_version.py', check : true).stdout().strip()
)

pymod = import('python')
python = pymod.find_installation('python3',
                                 required : true,
)

version = run_command(python, 'discover_version.py', '--full', check : true).stdout().strip().split()
version_dirty = version[0]
version_str = version[1]
version_hash = version[2]

# Those are requirements on *some older* systems, hence optional
cc = meson.get_compiler('c')
dl = cc.find_library('dl', required: false)
execinfo = cc.find_library('execinfo', required: false)

mugrid_dependencies = [dl, execinfo]

# This produces lots of Warning from Eigen3. Disabling for now.
# add_project_arguments('-Weffc++', language: 'cpp')

mpi = dependency('mpi', language : 'cpp', required : false)
if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    netcdf = dependency('pnetcdf', required : false)
    mugrid_dependencies += [mpi, netcdf]
    mpi_processes = ['1', '2']  # MPI processes to use for parallel tests
else
    netcdf = dependency('netcdf', required : false)
    mugrid_dependencies += [netcdf]
endif

if netcdf.found()
    mugrid_with_netcdf = true
    add_project_arguments('-DWITH_NETCDF_IO', language : ['c', 'cpp'])
else
    mugrid_with_netcdf = false
endif

eigen3 = dependency('eigen3', version : '>=3.4.0', required : true)
mugrid_dependencies += [eigen3]

message('  --------------------')
message('  muGrid configuration')
message('    MPI            : ' + (mpi.found() ? '*** YES ***' : 'no'))
if mpi.found()
    message('    Parallel NetCDF: ' + (netcdf.found() ? '*** YES ***' : 'no'))
else
    message('    Unidata NetCDF : ' + (netcdf.found() ? '*** YES ***' : 'no'))
endif
message('  --------------------')

subdir('src')
subdir('language_bindings')

test_timeout = 300  # timeout in seconds for long running tests

subdir('tests')
subdir('examples')
