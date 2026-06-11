"""
Tests for operator-reported ghost requirements and the `ghosts` argument of
GlobalFieldCollection, CartesianDecomposition and FFTEngine.
"""

import numpy as np
import pytest

import muGrid

# --- Operators report their ghost requirements ---


def test_laplace_ghost_requirement():
    op2 = muGrid.LaplaceOperator(2)
    assert op2.apply_ghost_requirement == ((1, 1), (1, 1))
    assert op2.transpose_ghost_requirement == ((1, 1), (1, 1))
    assert op2.ghost_requirement == ((1, 1), (1, 1))
    op3 = muGrid.LaplaceOperator(3)
    assert op3.ghost_requirement == ((1, 1, 1), (1, 1, 1))


def test_fem_gradient_ghost_requirement():
    op2 = muGrid.FEMGradientOperator(2, [0.1, 0.1])
    assert op2.apply_ghost_requirement == ((0, 0), (1, 1))
    # The scatter-style transpose writes into the same ghost buffers that
    # apply() reads, so the requirement is NOT mirrored
    assert op2.transpose_ghost_requirement == ((0, 0), (1, 1))
    assert op2.ghost_requirement == ((0, 0), (1, 1))
    op3 = muGrid.FEMGradientOperator(3, [0.1, 0.1, 0.1])
    assert op3.ghost_requirement == ((0, 0, 0), (1, 1, 1))


def test_generic_ghost_requirement():
    # Asymmetric stencil: offset [-1, 0], shape (3, 2)
    op = muGrid.GenericLinearOperator([-1, 0], np.ones((3, 2)))
    assert op.apply_ghost_requirement == ((1, 0), (1, 1))
    # The gather-style transpose mirrors the apply requirement
    assert op.transpose_ghost_requirement == ((1, 1), (1, 0))
    assert op.ghost_requirement == ((1, 1), (1, 1))


def test_isotropic_stiffness_ghost_requirement():
    op = muGrid.IsotropicStiffnessOperator(2, [0.1, 0.1])
    assert op.apply_ghost_requirement == ((1, 1), (1, 1))
    assert op.ghost_requirement == ((1, 1), (1, 1))


# --- Field collections and decompositions consume operators ---


def test_collection_from_operator():
    fc = muGrid.GlobalFieldCollection([8, 8], ghosts=muGrid.LaplaceOperator(2))
    assert fc.nb_ghosts_left == (1, 1)
    assert fc.nb_ghosts_right == (1, 1)


def test_collection_from_operator_list():
    # Elementwise maximum over several operators
    fc = muGrid.GlobalFieldCollection(
        [8, 8],
        ghosts=[
            muGrid.FEMGradientOperator(2, [0.1, 0.1]),  # (0, 0), (1, 1)
            muGrid.GenericLinearOperator([-1, 0], np.ones((3, 1))),
        ],
    )
    assert fc.nb_ghosts_left == (1, 0)
    assert fc.nb_ghosts_right == (1, 1)


def test_collection_from_int():
    fc = muGrid.GlobalFieldCollection([8, 8], ghosts=2)
    assert fc.nb_ghosts_left == (2, 2)
    assert fc.nb_ghosts_right == (2, 2)


def test_collection_from_explicit_pair():
    fc = muGrid.GlobalFieldCollection([8, 8], ghosts=((0, 0), (1, 1)))
    assert fc.nb_ghosts_left == (0, 0)
    assert fc.nb_ghosts_right == (1, 1)
    # Scalars in the pair are broadcast to all dimensions
    fc = muGrid.GlobalFieldCollection([8, 8], ghosts=(0, 1))
    assert fc.nb_ghosts_left == (0, 0)
    assert fc.nb_ghosts_right == (1, 1)


def test_collection_ghosts_conflict():
    with pytest.raises(ValueError):
        muGrid.GlobalFieldCollection(
            [8, 8], ghosts=1, nb_ghosts_left=[1, 1], nb_ghosts_right=[1, 1]
        )


def test_collection_ghosts_dim_mismatch():
    with pytest.raises(ValueError):
        muGrid.GlobalFieldCollection([8, 8], ghosts=muGrid.LaplaceOperator(3))


def test_decomposition_from_operator(comm):
    from NuMPI.Testing.Subdivision import suggest_subdivisions

    s = suggest_subdivisions(2, comm.size)
    laplace = muGrid.LaplaceOperator(2)
    decomposition = muGrid.CartesianDecomposition(
        comm, [16, 16], s, ghosts=laplace
    )
    assert decomposition.collection.nb_ghosts_left == (1, 1)
    assert decomposition.collection.nb_ghosts_right == (1, 1)
    # The operator must accept fields of this decomposition
    field = decomposition.real_field("in")
    result = decomposition.real_field("out")
    laplace.apply(field, result)


def test_fft_engine_from_operator(comm):
    laplace = muGrid.LaplaceOperator(2)
    engine = muGrid.FFTEngine([16, 16], comm, ghosts=laplace)
    # The engine may pad the ghost buffers further (e.g. for FFT alignment),
    # but must provide at least the operator's requirement
    left = engine.real_space_collection.nb_ghosts_left
    right = engine.real_space_collection.nb_ghosts_right
    assert all(g >= 1 for g in left)
    assert all(g >= 1 for g in right)
    field = engine.real_space_field("in")
    result = engine.real_space_field("out")
    laplace.apply(field, result)


# --- The runtime check matches the reported requirement ---


def test_apply_rejects_insufficient_ghosts():
    fc = muGrid.GlobalFieldCollection([8, 8])  # no ghosts
    field = fc.real_field("in")
    result = fc.real_field("out")
    laplace = muGrid.LaplaceOperator(2)
    with pytest.raises(RuntimeError, match="ghost"):
        laplace.apply(field, result)


def test_generic_apply_rejects_insufficient_ghosts():
    # One-sided ghosts suffice for apply but not for (mirrored) transpose
    op = muGrid.GenericLinearOperator([0, 0], np.ones((2, 2)))
    fc = muGrid.GlobalFieldCollection(
        [8, 8], ghosts=(op.apply_ghost_requirement)
    )
    field = fc.real_field("in")
    result = fc.real_field("out")
    op.apply(field, result)  # apply is fine
    with pytest.raises(RuntimeError, match="ghost"):
        op.transpose(field, result)
