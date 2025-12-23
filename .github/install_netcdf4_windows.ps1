# PowerShell script to install NetCDF on Windows for wheel building
# This script downloads and builds NetCDF-C from source using CMake

$ErrorActionPreference = "Stop"

Write-Host "Installing NetCDF on Windows..."
Write-Host "================================"

# Create a temporary directory for builds
$BUILD_DIR = Join-Path $env:TEMP "netcdf-build"
New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
Set-Location $BUILD_DIR

# Set installation prefix
$INSTALL_PREFIX = Join-Path $env:TEMP "netcdf-install"
New-Item -ItemType Directory -Force -Path $INSTALL_PREFIX | Out-Null

Write-Host "Install prefix: $INSTALL_PREFIX"
Write-Host "Build directory: $BUILD_DIR"

# Download and build NetCDF-C
$NETCDF_VERSION = "4.9.3"
Write-Host "Downloading NetCDF-C version $NETCDF_VERSION..."

$NETCDF_URL = "https://github.com/Unidata/netcdf-c/archive/refs/tags/v$NETCDF_VERSION.zip"
$NETCDF_ZIP = Join-Path $BUILD_DIR "netcdf.zip"

# Download NetCDF
Invoke-WebRequest -Uri $NETCDF_URL -OutFile $NETCDF_ZIP
Write-Host "Extracting NetCDF..."
Expand-Archive -Path $NETCDF_ZIP -DestinationPath $BUILD_DIR

# Build NetCDF
$NETCDF_SRC = Join-Path $BUILD_DIR "netcdf-c-$NETCDF_VERSION"
$NETCDF_BUILD = Join-Path $BUILD_DIR "build-netcdf"
New-Item -ItemType Directory -Force -Path $NETCDF_BUILD | Out-Null

Write-Host "Configuring NetCDF with CMake..."
Set-Location $NETCDF_BUILD

cmake -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" `
    -DBUILD_SHARED_LIBS=OFF `
    -DBUILD_TESTING=OFF `
    -DBUILD_TESTSETS=OFF `
    -DNETCDF_BUILD_UTILITIES=OFF `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DNETCDF_ENABLE_CDF5=ON `
    -DNETCDF_ENABLE_DAP=OFF `
    -DNETCDF_ENABLE_DAP2=OFF `
    -DNETCDF_ENABLE_DAP4=OFF `
    -DNETCDF_ENABLE_HDF5=OFF `
    -DNETCDF_ENABLE_PLUGINS=OFF `
    "$NETCDF_SRC"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}

Write-Host "Building NetCDF..."
cmake --build . --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Error "NetCDF build failed"
    exit 1
}

Write-Host "Installing NetCDF..."
cmake --install . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Error "NetCDF installation failed"
    exit 1
}

# Set environment variables for subsequent build steps
Write-Host "Setting environment variables..."
$env:CMAKE_PREFIX_PATH = "$INSTALL_PREFIX;$env:CMAKE_PREFIX_PATH"
$env:netCDF_DIR = Join-Path $INSTALL_PREFIX "lib\cmake\netCDF"

# For GitHub Actions, output environment variables
if ($env:GITHUB_ENV) {
    Write-Host "Adding to GITHUB_ENV..."
    Add-Content -Path $env:GITHUB_ENV -Value "CMAKE_PREFIX_PATH=$INSTALL_PREFIX;$env:CMAKE_PREFIX_PATH"
    Add-Content -Path $env:GITHUB_ENV -Value "netCDF_DIR=$env:netCDF_DIR"
}

# Also add to GITHUB_PATH so binaries can be found
if ($env:GITHUB_PATH) {
    $BIN_PATH = Join-Path $INSTALL_PREFIX "bin"
    if (Test-Path $BIN_PATH) {
        Add-Content -Path $env:GITHUB_PATH -Value $BIN_PATH
    }
}

Write-Host "NetCDF installation complete!"
Write-Host "netCDF_DIR: $env:netCDF_DIR"
Write-Host "CMAKE_PREFIX_PATH: $env:CMAKE_PREFIX_PATH"

# Verify installation
$NETCDF_LIB = Join-Path $INSTALL_PREFIX "lib\netcdf.lib"
if (Test-Path $NETCDF_LIB) {
    Write-Host "Verified: NetCDF library found at $NETCDF_LIB"
} else {
    Write-Warning "NetCDF library not found at expected location: $NETCDF_LIB"
}

# Clean up build directory but keep install
Set-Location $env:TEMP
Remove-Item -Recurse -Force $BUILD_DIR
Write-Host "Cleaned up build directory"
