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
curl -L https://download.gnome.org/sources/libxml2/2.12/libxml2-2.12.9.tar.xz | tar -Jx
cd libxml2-2.12.7
./configure --without-lzma --without-python --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
cd ..
curl -L https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/netcdf-c-4.9.2.tar.gz | tar -zx
cd netcdf-c-4.9.2
./configure --disable-hdf5 --disable-byterange --enable-static --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
