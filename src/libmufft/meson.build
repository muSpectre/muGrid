mufft_sources = [
    'muFFT',
    'version.cc',
    'derivative.cc',
    'fft_engine_base.cc',
    'fft_utils.cc',
    'fftw_engine.cc',
]

if mu_with_fftw3mpi
    mufft_sources += ['fftwmpi_engine.cc']
endif

if mu_with_pfft
    mufft_sources += ['pfft_engine.cc']
endif

mufft = shared_library(
    mufft_sources,
    include_directories: include_directories('..'),
    link_with: mugrid,
    dependencies: muspectre_dependencies
)
