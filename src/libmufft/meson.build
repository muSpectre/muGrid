incdirs = include_directories('..')

mufft_sources = [
    'version.cc',
    'derivative.cc',
    'fft_engine_base.cc',
    'fft_utils.cc',
    'pocketfft_engine.cc'
]

if mu_with_fftw3
    mufft_sources += ['fftw_engine.cc']
endif

if mu_with_fftw3mpi
    mufft_sources += ['fftwmpi_engine.cc']
endif

if mu_with_pfft
    mufft_sources += ['pfft_engine.cc']
endif

mufft = shared_library(
    'muFFT',
    mufft_sources,
    include_directories: [eigen3_incdir, incdirs],
    link_with: mugrid,
    dependencies: muspectre_dependencies,
    install: true  # needed, otherwise it is not copied to wheel
)
