#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_device_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    06 Jul 2026

@brief   Host tests for the Device abstraction, in particular the
         Device.from_string parser (inverse of the device_string property)

Copyright © 2026 Lars Pastewka

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.

------------------------------------------------------------------------------

Device is a value type, so these tests need no GPU: constructing
``Device.cuda(1)`` / ``Device.rocm(2)`` and parsing their string spellings is
pure host-side bookkeeping. ``Device.gpu()`` resolves to the compiled backend
at compile time, so it is validated against the runtime feature flags rather
than a fixed name.
"""

import pytest

import muGrid


def _gpu_type_name():
    """type_name that Device.gpu() resolves to on this build."""
    if muGrid.has_cuda:
        return "CUDA"
    if muGrid.has_rocm:
        return "ROCm"
    return "CPU"  # no GPU backend: gpu() falls back to the CPU device


def test_from_string_cpu():
    d = muGrid.Device.from_string("cpu")
    assert d.is_host
    assert not d.is_device
    assert d.type_name == "CPU"
    assert d == muGrid.Device.cpu()


@pytest.mark.parametrize("kind, type_name", [("cuda", "CUDA"), ("rocm", "ROCm")])
def test_from_string_explicit_backend(kind, type_name):
    # Bare form defaults to id 0.
    d0 = muGrid.Device.from_string(kind)
    assert d0.type_name == type_name
    assert d0.device_id == 0
    # Explicit id.
    d3 = muGrid.Device.from_string(f"{kind}:3")
    assert d3.type_name == type_name
    assert d3.device_id == 3


def test_from_string_gpu_alias():
    # "gpu" is an accepted input alias even though device_string never emits it.
    d = muGrid.Device.from_string("gpu:2")
    assert d.type_name == _gpu_type_name()
    if muGrid.has_gpu:
        assert d.device_id == 2


def test_from_string_is_case_insensitive():
    assert muGrid.Device.from_string("ROCm:1") == muGrid.Device.rocm(1)
    assert muGrid.Device.from_string("CUDA") == muGrid.Device.cuda(0)
    assert muGrid.Device.from_string("CPU") == muGrid.Device.cpu()


def test_from_string_roundtrips_device_string():
    # from_string is the inverse of the device_string property.
    for d in (
        muGrid.Device.cpu(),
        muGrid.Device.cuda(0),
        muGrid.Device.rocm(2),
    ):
        assert muGrid.Device.from_string(d.device_string) == d


@pytest.mark.parametrize(
    "spec",
    [
        "gpu:",       # empty id
        "rocm:x",     # non-numeric id
        "cuda:1abc",  # trailing garbage
        "cpu:0",      # cpu takes no id
        "foo",        # unknown kind
        "foo:0",      # unknown kind with id
        "",           # empty string
    ],
)
def test_from_string_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        muGrid.Device.from_string(spec)
