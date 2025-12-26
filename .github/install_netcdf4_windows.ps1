# PowerShell script to install NetCDF on Windows for wheel building
# This script builds zlib and NetCDF-C from source for static linking

$ErrorActionPreference = "Stop"

Write-Host "Environment:"
Write-Host "  CIBW_ARCHS: $env:CIBW_ARCHS"
Write-Host "  TEMP: $env:TEMP"

# Determine architecture
$arch = if ($env:CIBW_ARCHS -match "ARM64") { "ARM64" } else { "x64" }
Write-Host "Building for architecture: $arch"

# Use a fixed path for reliable CMake discovery
# This path must match CIBW_ENVIRONMENT_WINDOWS in wheels.yml
$installPrefix = "C:/netcdf-install"
$buildDir = "C:/netcdf-build"
Write-Host "Install prefix: $installPrefix"

# Create directories
New-Item -ItemType Directory -Force -Path $installPrefix | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# Download and build zlib (required by NetCDF)
$zlibVersion = "1.3.1"
Write-Host "Installing zlib-$zlibVersion"
$zlibUrl = "https://github.com/madler/zlib/releases/download/v$zlibVersion/zlib-$zlibVersion.tar.gz"
$zlibArchive = "$buildDir/zlib.tar.gz"
Invoke-WebRequest -Uri $zlibUrl -OutFile $zlibArchive

# Extract zlib
Push-Location $buildDir
tar -xzf zlib.tar.gz
Pop-Location

# Build zlib
$zlibBuildDir = "$buildDir/build-zlib"
$zlibSrcDir = "$buildDir/zlib-$zlibVersion"
New-Item -ItemType Directory -Force -Path $zlibBuildDir | Out-Null

Write-Host "Configuring zlib..."
cmake -S $zlibSrcDir -B $zlibBuildDir `
    -A $arch `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$installPrefix" `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DBUILD_SHARED_LIBS=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Error "zlib CMake configuration failed"
    exit 1
}

Write-Host "Building zlib..."
cmake --build $zlibBuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Error "zlib CMake build failed"
    exit 1
}

Write-Host "Installing zlib..."
cmake --install $zlibBuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Error "zlib CMake install failed"
    exit 1
}

# Download and build NetCDF-C
$netcdfVersion = "4.9.3"
Write-Host "Installing netcdf-c-$netcdfVersion"
$netcdfUrl = "https://github.com/Unidata/netcdf-c/archive/refs/tags/v$netcdfVersion.zip"
$netcdfArchive = "$buildDir/netcdf.zip"
Invoke-WebRequest -Uri $netcdfUrl -OutFile $netcdfArchive

# Extract using PowerShell (more reliable than tar on Windows)
Expand-Archive -Path $netcdfArchive -DestinationPath $buildDir -Force

# Build NetCDF
$netcdfBuildDir = "$buildDir/build-netcdf"
$netcdfSrcDir = "$buildDir/netcdf-c-$netcdfVersion"
New-Item -ItemType Directory -Force -Path $netcdfBuildDir | Out-Null

Write-Host "Configuring NetCDF..."
cmake -S $netcdfSrcDir -B $netcdfBuildDir `
    -A $arch `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$installPrefix" `
    -DCMAKE_PREFIX_PATH="$installPrefix" `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DBUILD_TESTING=OFF `
    -DBUILD_TESTSETS=OFF `
    -DNETCDF_BUILD_UTILITIES=OFF `
    -DNETCDF_ENABLE_CDF5=ON `
    -DNETCDF_ENABLE_DAP=OFF `
    -DNETCDF_ENABLE_DAP2=OFF `
    -DNETCDF_ENABLE_DAP4=OFF `
    -DNETCDF_ENABLE_HDF5=OFF `
    -DNETCDF_ENABLE_PLUGINS=OFF `
    -DNETCDF_ENABLE_BYTERANGE=OFF `
    -DNETCDF_ENABLE_LIBXML2=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed"
    exit 1
}

Write-Host "Building NetCDF..."
cmake --build $netcdfBuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake build failed"
    exit 1
}

Write-Host "Installing NetCDF..."
cmake --install $netcdfBuildDir --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake install failed"
    exit 1
}

# Clean up build files to save space
Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue

Write-Host "NetCDF installation complete at $installPrefix"
Write-Host "Contents of install directory:"
Get-ChildItem -Recurse $installPrefix | ForEach-Object { Write-Host $_.FullName }

# Set environment variable for subsequent build steps
Write-Host "Setting CMAKE_PREFIX_PATH environment variable..."
$currentPath = [Environment]::GetEnvironmentVariable("CMAKE_PREFIX_PATH", "Process")
if ($currentPath) {
    [Environment]::SetEnvironmentVariable("CMAKE_PREFIX_PATH", "$installPrefix;$currentPath", "Process")
} else {
    [Environment]::SetEnvironmentVariable("CMAKE_PREFIX_PATH", $installPrefix, "Process")
}
Write-Host "CMAKE_PREFIX_PATH: $([Environment]::GetEnvironmentVariable('CMAKE_PREFIX_PATH', 'Process'))"