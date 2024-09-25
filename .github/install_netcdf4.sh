# The variable HOMEBREW_REPOSITORY appears to be only present on ARM Macs, not Intel Macs
# HOMEBREW_NO_AUTO_UPDATE is present on both
if [ -z "$HOMEBREW_NO_AUTO_UPDATE" ]; then
    INSTALL_PREFIX=/usr/local
else
    brew install pkg-config
    INSTALL_PREFIX=$(which brew | sed 's,/bin/brew,,')
    # libxml2 pkg-config file is not in the default path on Intel Macs
    export PKG_CONFIG_PATH
fi
printenv

echo "Installing into prefix ${INSTALL_PREFIX}..."

XML2="libxml2-2.12.9"
curl -L https://download.gnome.org/sources/libxml2/2.12/${XML2}.tar.xz | tar -Jx
cd ${XML2}
./configure --without-lzma --without-python --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
cd ..
NETCDF=netcdf-c-4.9.2
curl -L https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/${NETCDF}.tar.gz | tar -zx
cd ${NETCDF}
./configure --disable-hdf5 --disable-byterange --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
