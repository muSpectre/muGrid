yum install -y libcurl-devel
curl https://downloads.unidata.ucar.edu/netcdf-c/4.9.2/netcdf-c-4.9.2.tar.gz | tar -zx
cd netcdf-c-4.9.2
./configure --disable-hdf5 --disable-shared --disable-libxml2 --with-pic
make
make install