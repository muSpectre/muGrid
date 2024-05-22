if [ -z "$HOMEBREW_REPOSITORY" ]; then
    INSTALL_PREFIX=/usr/local
else
    INSTALL_PREFIX=$HOMEBREW_REPOSITORY
    export PKG_CONFIG_PATH
fi
printenv
curl -L https://download.gnome.org/sources/libxml2/2.12/libxml2-2.12.7.tar.xz | tar -Jx
cd libxml2-2.12.7
./configure --without-python --prefix=$INSTALL_PREFIX
make
make install
cd ..
curl -L https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/netcdf-c-4.9.2.tar.gz | tar -zx
cd netcdf-c-4.9.2
./configure --disable-hdf5 --disable-byterange --disable-shared --with-pic --prefix=$INSTALL_PREFIX
make
make install
