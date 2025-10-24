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

XML2="libxml2-2.15.1"
echo "Installing ${XML2}"
curl -L https://download.gnome.org/sources/libxml2/2.15/${XML2}.tar.xz | tar -Jx
cd ${XML2}
./configure --build=$(echo $ARCHFLAGS | sed 's/-arch //') --without-python --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
cd ..

#NETCDF="netcdf-c-4.9.2"
#curl -L https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/${NETCDF}.tar.gz | tar -zx
NETCDF_VERSION="4.9.3"
echo "Installing netcdf-c-${NETCDF_VERSION}"
curl -L https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_VERSION}.tar.gz | tar -zx
cd netcdf-c-${NETCDF_VERSION}
./configure --build=$(echo $ARCHFLAGS | sed 's/-arch //') --disable-dap --disable-hdf5 --disable-nczarr --disable-byterange --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
