echo "uname -a is:"
uname -a
echo "============"

echo "Environment variables are:"
printenv
echo "=========================="

# The variable HOMEBREW_REPOSITORY appears to be only present on ARM Macs, not Intel Macs
# HOMEBREW_NO_AUTO_UPDATE is present on both
if [ -z "$INSTALL_PREFIX" ]; then
if [ -z "$HOMEBREW_NO_AUTO_UPDATE" ]; then
    INSTALL_PREFIX=/usr/local
else
    brew install pkg-config
    INSTALL_PREFIX=$(which brew | sed 's,/bin/brew,,')
    # libxml2 pkg-config file is not in the default path on Intel Macs
fi
fi

export PATH=${INSTALL_PREFIX}/bin:${PATH}
export PKG_CONFIG_PATH=${INSTALL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}

echo "Installing into prefix ${INSTALL_PREFIX}..."
echo "PATH is ${PATH}"
echo "PKG_CONFIG_PATH is ${PKG_CONFIG_PATH}"

XML2="libxml2-2.12.9"
curl -L https://download.gnome.org/sources/libxml2/2.12/${XML2}.tar.xz | tar -Jx
cd ${XML2}
cmake -DLIBXML2_WITH_LZMA=OFF -DLIBXML2_WITH_PYTHON=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .
make
make install
cd ..

NETCDF=netcdf-c-4.9.2
curl -L https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/${NETCDF}.tar.gz | tar -zx
cd ${NETCDF}
cmake -DENABLE_HDF5=OFF -DENABLE_CDF5=OFF -DENABLE_BYTERANGE=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .
make
make install
