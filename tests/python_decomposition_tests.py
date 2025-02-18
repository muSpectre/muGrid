import unittest
import numpy as np

import muGrid


class DecompositionCheck(unittest.TestCase):

    def get_nb_subdivisions(self, nb_processes: int):
        subdivision_setup = {
            1: (1,),
            2: (2,),
            4: (2, 2),
            8: (2, 2, 2),
        }
        if nb_processes in subdivision_setup:
            return subdivision_setup[nb_processes]
        else:
            raise NotImplementedError("Not planned for this number of processes.")

    def test_communicate_ghost(self):
        # Create a communicator
        try:
            from mpi4py import MPI
            comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            comm = muGrid.Communicator()
        nb_processes = comm.size

        # Create a Cartesian decomposition
        nb_subdivisions = self.get_nb_subdivisions(nb_processes)
        spatial_dim = len(nb_subdivisions)
        nb_pts_per_dim = 5
        nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)
        nb_ghost_left = np.full(spatial_dim, 1)
        nb_ghost_right = np.full(spatial_dim, 2)
        cart_decomp = muGrid.CartesianDecomposition(
            comm,
            nb_domain_grid_pts.tolist(),
            nb_subdivisions,
            nb_ghost_left.tolist(),
            nb_ghost_right.tolist(),
        )

        # Create a field for testing
        field_name = "test_field"
        field = cart_decomp.collection.real_field(field_name)

        # Create reference values
        global_coords = cart_decomp.global_coords
        weights = np.arange(spatial_dim) + 1
        ref_values = np.einsum("i, i...->...", weights, global_coords)

        # Fill the field, non-ghost with reference values, ghost with some other value
        nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts
        for index in np.ndindex(*nb_subdomain_grid_pts):
            is_not_ghost = all(
                idx >= nb_ghost_left[dim]
                and idx < nb_subdomain_grid_pts[dim] - nb_ghost_right[dim]
                for dim, idx in enumerate(index)
            )
            if is_not_ghost:
                field.s[(..., *index)] = ref_values[(..., *index)]
            else:
                field.s[(..., *index)] = -1

        # Communicate ghost cells
        cart_decomp.communicate_ghosts(field_name)

        # Check values at all grid points
        for index in np.ndindex(*nb_subdomain_grid_pts):
            self.assertEqual(
                field.s[(..., *index)],
                ref_values[(..., *index)],
                f"Mismatch at {index}",
            )


if __name__ == "__main__":
    unittest.main()
