#!/bin/bash
set -e

echo "uname -a is:"
uname -a
echo "============"

echo "Environment variables are:"
printenv
echo "=========================="

# Determine install prefix based on platform
OS_TYPE=$(uname -s)
echo "Detected OS: ${OS_TYPE}"

if [ -z "$INSTALL_PREFIX" ]; then
    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS - use Homebrew prefix
        brew install pkg-config cmake
        INSTALL_PREFIX=$(brew --prefix)
    else
        # Linux - use /usr/local
        INSTALL_PREFIX=/usr/local
    fi
fi

export PATH=${INSTALL_PREFIX}/bin:${PATH}
export PKG_CONFIG_PATH=${INSTALL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}

echo "Installing into prefix ${INSTALL_PREFIX}..."
echo "PATH is ${PATH}"
echo "PKG_CONFIG_PATH is ${PKG_CONFIG_PATH}"

# Build libxml2 from source (required for NetCDF on some platforms)
XML2_VERSION="2.15.1"
echo "Installing libxml2-${XML2_VERSION}"
curl -L https://download.gnome.org/sources/libxml2/2.15/libxml2-${XML2_VERSION}.tar.xz | tar -Jx
mkdir -p build-libxml2
cd build-libxml2
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLIBXML2_WITH_PYTHON=OFF \
    -DLIBXML2_WITH_TESTS=OFF \
    -DLIBXML2_WITH_PROGRAMS=OFF \
    ../libxml2-${XML2_VERSION}
cmake --build . --parallel
cmake --install .
cd ..
rm -rf build-libxml2 libxml2-${XML2_VERSION}

# Build NetCDF from source using CMake
NETCDF_VERSION="4.9.3"
echo "Installing netcdf-c-${NETCDF_VERSION}"
curl -L https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_VERSION}.tar.gz | tar -zx
mkdir -p build-netcdf
cd build-netcdf
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_TESTSETS=OFF \
    -DNETCDF_BUILD_UTILITIES=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DNETCDF_ENABLE_CDF5=ON \
    -DNETCDF_ENABLE_DAP=OFF \
    -DNETCDF_ENABLE_DAP2=OFF \
    -DNETCDF_ENABLE_DAP4=OFF \
    -DNETCDF_ENABLE_HDF5=OFF \
    -DNETCDF_ENABLE_PLUGINS=OFF \
    ../netcdf-c-${NETCDF_VERSION}
cmake --build . --parallel
cmake --install .
cd ..
rm -rf build-netcdf netcdf-c-${NETCDF_VERSION}

echo "NetCDF installation complete!"
echo "Checking pkg-config:"
pkg-config --modversion netcdf || echo "Warning: pkg-config netcdf not found"
pkg-config --libs netcdf || echo "Warning: pkg-config netcdf libs not found"
