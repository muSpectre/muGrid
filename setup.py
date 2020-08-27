# Copyright © 2019 Lars Pastewka
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Lesser Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
# Additional permission under GNU GPL version 3 section 7
#
# If you modify this Program, or any covered work, by linking or combining it
# with proprietary FFT implementations or numerical libraries, containing parts
# covered by the terms of those libraries' licenses, the licensors of this
# Program grant you additional permission to convey the resulting work.

# Adapted from: https://github.com/pybind/python_example
#               https://stackoverflow.com/questions/50938128/distutils-build-multiple-python-extension-modules-written-in-swig-that-share

import os
import re
import sys
import setuptools
import subprocess
from subprocess import PIPE

from distutils import sysconfig
from distutils.spawn import find_executable
from distutils.sysconfig import customize_compiler, get_config_var
from setuptools import setup, Extension
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext

# make sure _config_vars is initialized
get_config_var("LDSHARED")
from distutils.sysconfig import _config_vars as _CONFIG_VARS

###

disable_mpi = False
if '--disable-mpi' in sys.argv:
    index = sys.argv.index('--disable-mpi')
    sys.argv.pop(index)  # Removes the '--disable-mpi'
    disable_mpi = True

verbose = False
if '--verbose' in sys.argv:
    index = sys.argv.index('--verbose')
    sys.argv.pop(index)  # Removes the '--disable-mpi'
    verbose = True

###

fftw_info = {
    'name': 'fftw',
    'environ': 'FFTWDIR',
    'command': 'fftw-wisdom',
    'required_libraries': ['fftw3'],
    'mpi': False,
}

fftwmpi_info = {
    'name': 'fftwmpi',
    'environ': 'FFTWDIR',
    'command': 'fftw-wisdom',
    'required_libraries': ['fftw3_mpi'],
    'mpi': True,
    'define_macro': 'WITH_FFTWMPI'
}

pfft_info = {
    'name': 'pfft',
    'environ': 'PFFTDIR',
    'required_libraries': ['pfft'],
    'mpi': True,
    'define_macro': 'WITH_PFFT'
}

###

def get_version_from_git():
    """
    Discover muSpectre version from git repository.
    """
    git_describe = subprocess.run(
        ['git', 'describe', '--tags', '--dirty', '--always'],
        stdout=subprocess.PIPE)
    if git_describe.returncode != 0:
        raise RuntimeError('git execution failed')
    version = git_describe.stdout.decode('latin-1').strip()
    git_hash = subprocess.run(
        ['git', 'show', '-s', '--format=%H'],
        stdout=subprocess.PIPE)
    if git_hash.returncode != 0:
        raise Runtimeerror('git execution failed')
    hash = git_hash.stdout.decode('latin-1').strip()

    if verbose:
        print('GIT Version detected:', version)

    return version.endswith('dirty'), version, hash


def get_version_from_cc(fn):
    text = open(fn, 'r').read()
    dirty = bool(re.search('constexpr bool git_dirty{(true|false)};',
                           text).group(1))
    version = re.search(
        'constexpr char git_describe\[\]{"([A-Za-z0-9_.-]*)"};', text
    ).group(1)
    hash = re.search('constexpr char git_hash\[\]{"([A-Za-z0-9_.-]*)"};',
                     text).group(1)

    if verbose:
        print('Version contained in version.cc:', version)

    return dirty, version, hash


def detect_library(info):
    if verbose:
        print("Detecting location of library '{}'".format(info['name']))

    found = False
    root = None
    if 'environ' in info and info['environ'] in os.environ:
        if verbose:
            print("  * Looking for environment variable '{}'"
                .format(info['environ']))
        root = os.environ[info['environ']]
        found = True
    if not found:
        libname = info['required_libraries'][0]
        if verbose:
            print("  * Attempting to load library '{}'".format(libname))
        import ctypes.util
        full_libname = ctypes.util.find_library(libname)
        if full_libname is not None:
            # This mechanism does not give us the location of the
            # library
            found = True
    if not found and 'command' in info:
        if verbose:
            print("  * Looking for executable '{}'"
                  .format(info['command']))
        command_path = find_executable(info['command'])
        if command_path is not None:
            root = os.path.abspath(os.path.dirname(command_path) +
                                   '/../lib')
        if verbose:
            print("  * Attempting to load library '{}' in path '{}'"
                .format(libname, root))
        import ctypes
        found = True
        try:
            loaded_lib = ctypes.CDLL('{}/fftw3_mpi.so')
        except OSError:
            found = False
    include_dirs = []
    libraries = []
    library_dirs = []
    if found:
        if root is not None and (root.endswith('/bin') or \
            root.endswith('/lib')):
            root = root[:-4]

        if verbose:
            if root is None:
                print("  * Detected library '{}' in standard library "
                      "search path".format(info['name']))
            else:
                print("  * Detected library '{}' in path '{}'"
                    .format(info['name'], root))

        for lib in info['required_libraries']:
            if root is not None:
                include_dirs += ['{}/include'.format(root)]
                library_dirs += ['{}/lib'.format(root)]
            libraries += [lib]

    else:
        if verbose:
            print("  ! Could not detect library '{}'"
                  .format(info['name']))
        return None

    return include_dirs, libraries, library_dirs


def get_eigen_include(eigen_version='3.3.5'):
    """
    Helper function to download and install eigen and return include
    path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    eigen_path = '{}/depend/eigen-{}'.format(root, eigen_version)
    if not os.path.exists(eigen_path):
        os.makedirs(eigen_path, exist_ok=True)
        os.system(
            'curl -L https://gitlab.com/libeigen/eigen/-/archive/{0}/eigen-{0}'
            '.tar.bz2 | tar -jx -C {1} --strip-components 1'
            .format(eigen_version, eigen_path))
    return(eigen_path)


def get_pybind11_include(pybind11_version='2.2.3'):
    """
    Helper function to download and install pybind and return include
    path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    pybind11_path = '{}/depend/pybind11-{}'.format(root, pybind11_version)
    if not os.path.exists(pybind11_path):
        os.makedirs(pybind11_path, exist_ok=True)
        os.system(
            'curl -L https://github.com/pybind/pybind11/archive/v{}'
            '.tar.gz | tar -zx -C {} --strip-components 1'
            .format(pybind11_version, pybind11_path))
    return('{}/include'.format(pybind11_path))

###

def _customize_compiler_for_shlib(compiler):
    if sys.platform == "darwin":
        # building .dylib requires additional compiler flags on OSX; here we
        # temporarily substitute the pyconfig.h variables so that distutils'
        # 'customize_compiler' uses them before we build the shared libraries.
        tmp = _CONFIG_VARS.copy()
        try:
            _CONFIG_VARS['LDSHARED'] = (
                "gcc -Wl,-x -dynamiclib -undefined dynamic_lookup")
            _CONFIG_VARS['SHLIB_SUFFIX'] = ".dylib"
            customize_compiler(compiler)
        finally:
            _CONFIG_VARS.clear()
            _CONFIG_VARS.update(tmp)
    else:
        customize_compiler(compiler)

###

if verbose:
    print('=== DETECTING FFT LIBRARIES ===')

mugrid_sources = [
    'src/libmugrid/ccoord_operations.cc',
    'src/libmugrid/exception.cc',
    'src/libmugrid/field.cc',
    'src/libmugrid/field_collection.cc',
    'src/libmugrid/field_collection_global.cc',
    'src/libmugrid/field_collection_local.cc',
    'src/libmugrid/field_map.cc',
    'src/libmugrid/field_typed.cc',
    'src/libmugrid/grid_common.cc',
    'src/libmugrid/raw_memory_operations.cc',
    'src/libmugrid/state_field.cc',
    'src/libmugrid/state_field_map.cc',
    'src/libmugrid/units.cc',
]
pymugrid_sources = [
    'language_bindings/libmugrid/python/bind_py_module.cc',
    'language_bindings/libmugrid/python/bind_py_common.cc',
    'language_bindings/libmugrid/python/bind_py_field.cc',
    'language_bindings/libmugrid/python/bind_py_field_collection.cc',
]
mufft_sources = [
    'src/libmufft/version.cc',
    'src/libmufft/fft_engine_base.cc',
    'src/libmufft/fft_utils.cc',
    'src/libmufft/derivative.cc',
]
pymufft_sources = [
    'language_bindings/libmufft/python/bind_py_module.cc',
    'language_bindings/libmufft/python/bind_py_common.cc',
    'language_bindings/libmufft/python/bind_py_derivatives.cc',
    'language_bindings/libmufft/python/bind_py_communicator.cc',
    'language_bindings/libmufft/python/bind_py_fftengine.cc',
]

macros = []
include_dirs = [get_eigen_include(), # Path to pybind11 headers
                get_pybind11_include(), # Path to Eigen headers
                'src']
fft_libraries = []
fft_library_dirs = []

# Did we manually disable MPI?
if disable_mpi:
    mpi = False
    print('MPI disabled with command line argument --disable-mpi.')
else:
    # We only enable MPI if mpicc is in the path
    try:
        mpicc = os.environ['MPICC']
    except KeyError:
        mpicc = 'mpicc'
    try:
        mpicxx = os.environ['MPICXX']
    except KeyError:
        mpicxx = 'mpicxx'

    # Test if we can execute mpicc and mpicxx
    try:
        mpicc_successful = subprocess.run(
            [mpicc, '--version'],
            stdout=PIPE, stderr=PIPE).returncode == 0
        mpicc_successful &= subprocess.run(
            [mpicxx, '--version'],
            stdout=PIPE, stderr=PIPE).returncode == 0
    except FileNotFoundError:
        mpicc_successful = False
    if not mpicc_successful:
        print('MPI disabled because MPI compiler wrappers were not '
              'found or could not be executed.')
    mpi = mpicc_successful

has_mpi_enabled_fft = False
for info, _sources in [(fftw_info, ['src/libmufft/fftw_engine.cc']),
                       (fftwmpi_info, ['src/libmufft/fftwmpi_engine.cc']),
                       (pfft_info, ['src/libmufft/pfft_engine.cc'])]:
    lib = detect_library(info)
    if lib is not None:
        _include_dirs, _libraries, _library_dirs = lib
        # Only include this library if it is serial or if MPI is
        # enabled
        if not info['mpi'] or mpi:
            mufft_sources += _sources
            if 'define_macro' in info:
                macros += [(info['define_macro'], None)]
            include_dirs += _include_dirs
            fft_libraries += _libraries
            fft_library_dirs += _library_dirs
        if info['mpi']:
            has_mpi_enabled_fft = True

if mpi and not has_mpi_enabled_fft:
    print('MPI disabled because no MPI-enabled FFT library was '
          'found.')
    mpi = False

if mpi:
    print('ENABLING MPI: At least one of the FFT libraries is '
          'MPI-parallel and the MPI compiler wrappers were found '
          'in the path.')
    print('(You can specify the compiler wrappers through the MPICC '
          'and MPICXX environment variables.)')

    macros += [('WITH_MPI', None)]
    # FIXME! This is a brute-force override.
    try:
        os.environ['CC'] = os.environ['MPICC']
    except KeyError:
        os.environ['CC'] = 'mpicc'
    try:
        os.environ['CXX'] = os.environ['MPICXX']
    except:
        os.environ['CXX'] = 'mpicxx'
    if verbose:
        print('  * C-compiler: {}'.format(os.environ['CC']))
        print('  * C++-compiler: {}'.format(os.environ['CXX']))
    mufft_sources += ['src/libmufft/communicator.cc']

# extra_link_args is required to search for shared libraries relative
# to the library's location. Specifically, muGrid.so and muFFT.so are
# placed in the install directory under 'site-packages', and the
# Python wrappers _muGrid and _muFFT need be able to find those.
extra_link_args = []
if sys.platform != 'darwin':
    extra_link_args += ['-Wl,-rpath,${ORIGIN}']

# We compile two shared libraries, libmuGrid.so and libmuFFT.so.
# These libraries do not contain the Python interface.

ext_libraries = [
    ('muGrid',
     dict(sources=mugrid_sources,
          macros=macros,
          include_dirs=include_dirs,
          language='c++',
          extra_link_args=extra_link_args)
    ),
    ('muFFT',
     dict(sources=mufft_sources,
          macros=macros,
          include_dirs=include_dirs,
          libraries=fft_libraries + ['muGrid'],
          library_dirs=fft_library_dirs,
          language='c++',
          extra_link_args=extra_link_args)
    ),
]

# We compile two Python interfaces, _muGrid and _muFFT that use the
# above shared libraries.

ext_modules = [
    Extension(
        '_muGrid',
        sources=pymugrid_sources,
        define_macros=macros,
        include_dirs=include_dirs,
        libraries=['muGrid'],
        extra_link_args=extra_link_args,
        language='c++',
    ),
    Extension(
        '_muFFT',
        sources=pymufft_sources,
        define_macros=macros,
        include_dirs=include_dirs,
        libraries=['muGrid', 'muFFT'],
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 '
                           'support is needed!')


def compiler_options(compiler, version):
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-Werror', '-Wno-deprecated-declarations'],
    }

    ct = compiler.compiler_type
    opts = c_opts.get(ct, [])

    if ct == 'unix':
        opts.append('-DVERSION_INFO="%s"' % version)
        opts.append(cpp_flag(compiler))
    elif ct == 'msvc':
        opts.append('/DVERSION_INFO=\\"%s\\"' % version)
    return opts


def linker_options(compiler):
    opts = [cpp_flag(compiler)] # Not sure if this is necessary
    if sys.platform == 'darwin':
        opts += ['-stdlib=libc++']
    return opts


class build_ext_custom(build_ext):
    """
    A custom build extension for adding compiler-specific options.
    """
    def build_extension(self, ext):
        sources = ext.sources

        ext_path = self.get_ext_fullpath(ext.name)
        language = ext.language or self.compiler.detect_language(sources)

        copts = compiler_options(self.compiler,
                                 self.distribution.get_version())
        extra_args = (ext.extra_compile_args or []) + copts

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        objects = self.compiler.compile(sources,
                                        output_dir=self.build_temp,
                                        macros=macros,
                                        include_dirs=ext.include_dirs,
                                        debug=self.debug,
                                        extra_postargs=extra_args,
                                        depends=ext.depends)

        self._built_objects = objects[:]

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        lopts = linker_options(self.compiler)
        extra_args = (ext.extra_link_args or []) + lopts

        self.compiler.link_shared_object(
            objects, ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language)


class build_clib_dyn(build_clib):
    """
    A custom build extension for adding compiler-specific options.
    """
    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('build_lib', 'build_clib'),
                                   ('build_temp', 'build_temp'),
                                   ('compiler', 'compiler'),
                                   ('debug', 'debug'),
                                   ('force', 'force'))
        self.libraries = self.distribution.libraries
        if self.libraries:
            self.check_library_list(self.libraries)
        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

    def build_libraries(self, libraries):
        _customize_compiler_for_shlib(self.compiler)
        copts = compiler_options(self.compiler,
                                 self.distribution.get_version())
        lopts = linker_options(self.compiler)
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % lib_name)
            sources = list(sources)

            language = build_info.get('language') or \
                self.compiler.detect_language(sources)

            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')

            extra_args = (build_info.get('extra_compile_args') or []) + \
                         copts

            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            extra_postargs=extra_args,
                                            debug=self.debug)
            library_dirs = (build_info.get('library_dirs') or []) + \
                [self.build_clib]

            library_filename = self.compiler.library_filename(
                lib_name, lib_type="shared")

            if sys.platform == 'darwin':
                extra_args = (build_info.get('extra_link_args') or []) + \
                    lopts + ['-Wl,-install_name,@loader_path/{}'
                             .format(library_filename)]

            self.compiler.link_shared_object(
                objects,
                library_filename,
                libraries=build_info.get('libraries'),
                library_dirs=library_dirs,
                output_dir=self.build_clib,
                extra_postargs=extra_args,
                debug=self.debug,
                target_lang=language)

requirements = ['numpy']
if mpi:
    requirements += ['mpi4py']

### Discover version and write version.cc

try:
    # Get version from git. If we get this from git, then we need to
    # check whether we have to refresh the version.cc file.
    dirty, version, hash = get_version_from_git()
    try:
        cc_dirty, cc_version, cc_hash = \
            get_version_from_cc('src/libmufft/version.cc')
        cc_found = True
    except:
        cc_found = False
    if not cc_found or cc_hash != hash or cc_version != version:
        # Write a new version.cc file
        if verbose:
            print('  * Writing new src/libmufft/version.cc')
        open('src/libmufft/version.cc', 'w').write(
            open('src/libmufft/version.cc.skeleton').read()
                .replace('@GIT_IS_DIRTY@', 'true' if dirty else 'false')
                .replace('@GIT_COMMIT_DESCRIBE@', version)
                .replace('@GIT_HEAD_SHA1@', hash))
except:
    # Detection via git failed. Get version from version.cc file.
    try:
        dirty, version, hash = \
            get_version_from_cc('src/libmufft/version.cc')
    except:
        raise RuntimeError('Version detection failed. This is not a '
                           'git repository and src/libmufft/version.cc '
                           'does not exist.')

setup(
    name='muFFT',
    version=version,
    author='Till Junge',
    author_email='till.junge@altermail.ch',
    url='https://gitlab.com/muspectre/muspectre',
    description='muFFT is a wrapper for common FFT libraries with '
                'support for MPI parallelization',
    long_description='',
    packages = ['muFFT', 'muGrid'],
    package_dir = {
        'muFFT': 'language_bindings/libmufft/python/muFFT',
        'muGrid': 'language_bindings/libmugrid/python/muGrid'
    },
    libraries=ext_libraries,
    ext_modules=ext_modules,
    cmdclass={'build_clib': build_clib_dyn,
              'build_ext': build_ext_custom},
    zip_safe=False,
    test_suite='tests',
    setup_requires=requirements,
    install_requires=requirements
)
