# PowerShell script to install NetCDF on Windows for wheel building

$arch = if ($env:CIBW_ARCHS -match "ARM64") { "arm64-windows" } else { "x64-windows" }
vcpkg install netcdf-c:$arch