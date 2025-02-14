import unittest
import numpy as np

import muGrid
from mpi4py import MPI


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
        comm = MPI.COMM_WORLD
        nb_processes = comm.size
        nb_subdivisions = self.get_nb_subdivisions(nb_processes)

        spatial_dim = len(nb_subdivisions)
        nb_pts_per_dim = 5
        nb_domain_grid_pts = np.full(spatial_dim, nb_pts_per_dim)

        ref_values = np.reshape(np.arange(nb_pts_per_dim**spatial_dim), nb_domain_grid_pts)

        nb_ghost_left = np.full(spatial_dim, 1)
        nb_ghost_right = np.full(spatial_dim, 2)
        cart_decomp = muGrid.CartesianDecomposition(
            muGrid.Communicator(comm),
            nb_domain_grid_pts.tolist(),
            nb_subdivisions,
            nb_ghost_left.tolist(),
            nb_ghost_right.tolist(),
        )

        field_name = "test_field"
        field = cart_decomp.collection.real_field(field_name)

        subdomain_locations = cart_decomp.subdomain_locations
        nb_subdomain_grid_pts = cart_decomp.nb_subdomain_grid_pts

        # Fill non-ghost cells
        for index in np.ndindex(*nb_subdomain_grid_pts):
            is_not_ghost = all(
                idx >= nb_ghost_left[dim]
                and idx < nb_subdomain_grid_pts[dim] - nb_ghost_right[dim]
                for dim, idx in enumerate(index)
            )
            if is_not_ghost:
                global_coords = (
                    subdomain_locations + np.array(index)
                ) % nb_domain_grid_pts
                full_index = (..., *index)
                field.s[full_index] = ref_values[global_coords]

        # Communicate ghost cells
        cart_decomp.communicate_ghosts(field_name)

        # Validate ghost cells
        for index in np.ndindex(*nb_subdomain_grid_pts):
            is_ghost = any(
                idx < nb_ghost_left[dim]
                or idx >= nb_subdomain_grid_pts[dim] - nb_ghost_right[dim]
                for dim, idx in enumerate(index)
            )
            if is_ghost:
                global_coords = (
                    subdomain_locations + np.array(index)
                ) % nb_domain_grid_pts
                full_index = (..., *index)
                self.assertEqual(
                    field.s[full_index],
                    ref_values[global_coords],
                    f"Mismatch at {index}",
                )


if __name__ == "__main__":
    unittest.main()
