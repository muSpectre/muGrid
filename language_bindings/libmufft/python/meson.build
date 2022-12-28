mufft_python_sources = [
    'bind_py_module.cc',
    'bind_py_common_mufft.cc',
    'bind_py_derivatives.cc',
    'bind_py_fftengine.cc'
]

mufft_python = python.extension_module(
    '_muFFT', # Name of the module
    mufft_python_sources,
    install: true,  # Install it
    include_directories: [eigen3_incdir, pybind11_incdir, include_directories('../../../src')],
    link_with: [mugrid, mufft],
    dependencies: [muspectre_dependencies, python3_dependency]
)

subdir('muFFT')
