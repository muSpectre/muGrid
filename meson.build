project(
    'muSpectre', # Project name
    'c', 'cpp', # Project type. We need a C and C++ compiler.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for std::optional
    version: run_command('python3', 'discover_version.py', check: true).stdout().strip()
)

pymod = import('python')
python = pymod.find_installation('python3',
    required: true,
)

version = run_command(python, 'discover_version.py', '--full', check: true).stdout().strip().split()
version_dirty = version[0]
version_str = version[1]
version_hash = version[2]

eigen3_incdir = include_directories('external/eigen3')

fftw3 = dependency('fftw3', required: false)
if fftw3.found()
    mu_with_fftw3 = true
    message('muSpectre FFTW: *** YES ***')
    add_global_arguments('-DWITH_FFTW', language : ['c', 'cpp'])
else
    mu_with_fftw3 = false
    message('muSpectre FFTW: no')
endif

# Those are requirements on *some* systems, hence optional
cc = meson.get_compiler('c')
dl = cc.find_library('dl', required: false)
execinfo = cc.find_library('execinfo', required: false)

muspectre_dependencies = [dl, execinfo, fftw3]

# This produces lots of Warning from Eigen3. Disabling for now.
# add_global_arguments('-Weffc++', language: 'cpp')

mpi = dependency('mpi', language: 'cpp', required: false)
if mpi.found()
    mu_with_mpi = true
    message('muSpectre MPI: *** YES ***')
    add_global_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    netcdf = dependency('pnetcdf', required: false)
    muspectre_dependencies += [mpi, netcdf]
    fftw3mpi_incdir = include_directories('/usr/include')
    fftw3mpi = cc.find_library(
        'fftw3_mpi',
        dirs: ['/usr/lib', '/usr/lib/x86_64-linux-gnu'])
    if fftw3mpi.found() and cc.has_header('fftw3-mpi.h')
        message('muSpectre FFTW3 MPI: *** YES ***')
        mu_with_fftw3mpi = true
        add_global_arguments('-DWITH_FFTWMPI', language : ['c', 'cpp'])
        muspectre_dependencies += [fftw3mpi]
    else
        message('muSpectre FFTW3 MPI: no')
        mu_with_fftw3mpi = false
    endif
    pfft = cc.find_library('pfft', required: false)
    if pfft.found() and cc.has_header('pfft.h')
        message('muSpectre PFFT: *** YES ***')
        mu_with_pfft = true
        add_global_arguments('-DWITH_PFFT', language : ['c', 'cpp'])
        muspectre_dependencies += [pfft]
    else
        message('muSpectre PFFT: no')
        mu_with_pfft = false
    endif
    mpi_processes = ['1', '2']  # MPI processes to use for parallel tests
else
    mu_with_mpi = false
    mu_with_fftw3mpi = false
    mu_with_pfft = false
    message('muSpectre MPI: no')
    netcdf = dependency('netcdf', required: false)
    muspectre_dependencies += [netcdf]
endif

if netcdf.found()
    mugrid_with_netcdf = true
    message('muSpectre NetCDF I/O: *** YES ****')
    add_global_arguments('-DWITH_NETCDF_IO', language : ['c', 'cpp'])
else
    mugrid_with_netcdf = false
    message('muSpectre NetCDF I/O: no')
endif

subdir('src')
subdir('language_bindings')

test_timeout = 300  # timeout in seconds for long running tests

subdir('tests')
