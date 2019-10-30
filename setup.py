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

from distutils.spawn import find_executable
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import setuptools

__version__ = '0.1'

###

if os.uname()[0] == 'Darwin':
    lib_extension = 'dylib'
else:
    lib_extension = 'so'

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

def detect_library(info):
    print("Detecting location of library '{}'".format(info['name']))

    found = False
    root = None
    if 'environ' in info and info['environ'] in os.environ:
        print("  * Looking for environment variable '{}'".format(info['environ']))
        root = os.environ[info['environ']]
        found = True
    if not found:
        libname = info['required_libraries'][0]
        print("  * Attempting to load library '{}'".format(libname))
        import ctypes.util
        full_libname = ctypes.util.find_library(libname)
        if full_libname is not None:
            # This mechanism does not give us the location of the library
            found = True
    if not found and 'command' in info:
        print("  * Looking for executable '{}'".format(info['command']))
        command_path = find_executable(info['command'])
        if command_path is not None:
            root = os.path.abspath(os.path.dirname(command_path) + '/../lib')
        print("  * Attempting to load library '{}' in path '{}'".format(libname, root))
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
        if root is not None and (root.endswith('/bin') or root.endswith('/lib')):
            root = root[:-4]

        if root is None:
            print("  * Detected library '{}' in standard library search path".format(info['name']))
        else:
            print("  * Detected library '{}' in path '{}'".format(info['name'], root))

        for lib in info['required_libraries']:
            if root is not None:
                include_dirs += ['{}/include'.format(root)]
                library_dirs += ['{}/lib'.format(root)]
            libraries += [lib]

    else:
        print("  ! Could not detect library '{}'".format(info['name']))
        return None

    return include_dirs, libraries, library_dirs


def get_eigen_include(eigen_version='3.3.5'):
    """Helper function to download and install eigen and return include path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    eigen_path = '{}/depend/eigen-{}'.format(root, eigen_version)
    if not os.path.exists(eigen_path):
        os.makedirs(eigen_path, exist_ok=True)
        os.system('curl -L http://bitbucket.org/eigen/eigen/get/{}.tar.bz2 | tar -jx -C {} --strip-components 1'.format(eigen_version, eigen_path))
    return(eigen_path)


def get_pybind11_include(pybind11_version='2.2.3'):
    """Helper function to download and install pybind and return include path.
    """
    root = os.path.abspath(os.path.dirname(__file__))
    pybind11_path = '{}/depend/pybind11-{}'.format(root, pybind11_version)
    if not os.path.exists(pybind11_path):
        os.makedirs(pybind11_path, exist_ok=True)
        os.system('curl -L https://github.com/pybind/pybind11/archive/v{}.tar.gz | tar -zx -C {} --strip-components 1'.format(pybind11_version, pybind11_path))
    return('{}/include'.format(pybind11_path))    

###

print('=== DETECTING FFT LIBRARIES ===')

sources = ['language_bindings/libmufft/python/bind_py_module.cc',
           'language_bindings/libmufft/python/bind_py_common.cc',
           'language_bindings/libmufft/python/bind_py_communicator.cc',
           'language_bindings/libmufft/python/bind_py_fftengine.cc',
           'src/libmufft/fft_engine_base.cc',
           'src/libmufft/communicator.cc',
           'src/libmufft/fft_utils.cc']
macros = []
include_dirs = [get_eigen_include(), # Path to pybind11 headers
                get_pybind11_include(), # Path to Eigen headers
                'src']
libraries = []
library_dirs = []

mpi = False
for info, _sources in [(fftw_info, ['src/libmufft/fftw_engine.cc']),
                       (fftwmpi_info, ['src/libmufft/fftwmpi_engine.cc']),
                       (pfft_info, ['src/libmufft/pfft_engine.cc'])]:
    lib = detect_library(info)
    if lib is not None:
        _include_dirs, _libraries, _library_dirs = lib
        sources += _sources
        if 'define_macro' in info:
            macros += [(info['define_macro'], None)]
        include_dirs += _include_dirs
        libraries += _libraries
        library_dirs += _library_dirs
        if 'mpi' in info:
            mpi = mpi or info['mpi']
if mpi:
    print('At least one of the FFT libraries is MPI-parallel. Using the MPI compiler wrapper.')
    print('(You can specify the compiler wrapper through the MPICC and MPICXX environment variables.)')
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
    print('  * C-compiler: {}'.format(os.environ['CC']))
    print('  * C++-compiler: {}'.format(os.environ['CXX']))

ext_modules = [
    Extension(
        '_muFFT',
        sources=sources,
        define_macros=macros,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
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
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        opts.append('-Werror')
        opts.append('-Wno-deprecated-declarations')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='muFFT',
    version=__version__,
    author='Till Junge',
    author_email='till.junge@altermail.ch',
    url='https://gitlab.com/muspectre/muspectre',
    description='muFFT is a wrapper for common FFT libraries with support '
                'MPI parallelization',
    long_description='',
    packages = ['muFFT'],
    package_dir = {'muFFT': 'language_bindings/libmufft/python/muFFT'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    test_suite='tests'
)
