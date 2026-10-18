"""The CUDA and HIP builds compile the same GPU sources.

``src/libmugrid/CMakeLists.txt`` lists the GPU sources twice, once per
backend, and in each branch twice more: appended to the library sources, and
marked with the backend's ``LANGUAGE``. The GPU CI builds only CUDA, so a file
missing from the HIP branch goes unnoticed until someone builds for ROCm --
which is how 1.5.0 shipped without ``linalg/block_matvec_gpu.cc`` on HIP and
failed to import there with an undefined symbol. These checks read the
CMakeLists.txt, so they need no GPU and run in every CI job.
"""

import re
from pathlib import Path

import pytest

CMAKELISTS = (Path(__file__).resolve().parents[1]
              / "src" / "libmugrid" / "CMakeLists.txt")

# Sources that exist for one backend only, by design.
BACKEND_SPECIFIC = {
    "CUDA": {"fft/cufft_backend.cc"},
    "HIP": {"fft/rocfft_backend.cc"},
}


def _branches():
    """``{backend: (appended sources, sources marked with its LANGUAGE)}``."""
    if not CMAKELISTS.exists():
        pytest.skip(f"source tree not available ({CMAKELISTS})")
    text = CMAKELISTS.read_text()
    match = re.search(
        r"^if\(MUGRID_ENABLE_CUDA\)\n(.*?)^elseif\(MUGRID_ENABLE_HIP\)\n"
        r"(.*?)^endif\(\)", text, re.MULTILINE | re.DOTALL)
    assert match, "GPU source block not found in src/libmugrid/CMakeLists.txt"
    branches = {}
    for backend, body in zip(("CUDA", "HIP"), match.groups()):
        appended = set(re.findall(
            r"list\(APPEND MUGRID_SOURCES (\S+)\)", body))
        marked = re.search(
            r"set_source_files_properties\((.*?)PROPERTIES LANGUAGE (\w+)",
            body, re.DOTALL)
        assert marked, f"no set_source_files_properties in the {backend} branch"
        assert marked.group(2) == backend
        language = set(re.findall(
            r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/(\S+)", marked.group(1)))
        branches[backend] = appended, language
    return branches


def test_cuda_and_hip_compile_the_same_gpu_sources():
    branches = _branches()
    common = {backend: appended - BACKEND_SPECIFIC[backend]
              for backend, (appended, _) in branches.items()}
    assert common["CUDA"] - common["HIP"] == set(), "missing from the HIP branch"
    assert common["HIP"] - common["CUDA"] == set(), "missing from the CUDA branch"


@pytest.mark.parametrize("backend", ["CUDA", "HIP"])
def test_every_gpu_source_is_compiled_as_gpu_code(backend):
    """A source appended but not marked is compiled as plain C++, which fails
    obscurely; one marked but not appended is not compiled at all."""
    appended, language = _branches()[backend]
    assert appended - language == set(), f"not marked LANGUAGE {backend}"
    assert language - appended == set(), "marked but not appended"
