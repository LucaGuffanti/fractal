#
# Here we will take a look at different iteration functions!
# This allows us to get more interesting results.
# Increasing the power of the iteration function increases the symmetry

from typing import Any

import core.MPIJulia as mpiJulia
import core.SequentialJulia as seq
import utils.Renderer as renderer
from mpi4py import MPI

import numpy as np

if __name__ == "__main__":

    # def mandelbrot_iteration(
    #     z: np.ndarray[int, np.complex128],
    #     to_compute: np.ndarray[Any, np.dtype[bool]],
    #     c
    # ):
    #     z[to_compute] = (z[to_compute])**2 + c[to_compute]
    #     return z
    

    # # ======================================================
    # # Come on, let's build more interestig fractals! 
    # # First, a small resume.
    # # ======================================================
    
    # size = 2000
    # mpiSet = mpiJulia.MPIJulia(
    #      size, size
    # )
    
    # fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 500, -1.476)
    # if mpiSet.rank == 0:
    #     res = np.concatenate(fractal, axis=1)
    #     renderer.save_fractal(res, "./graphics/advanced_fractal_1.jpg")

    # # ======================================================
    # # So, let's try changing the iteration function... If
    # # we just increase the power of the iteration we will
    # # get more symmetry
    # # ======================================================
    
    # def my_first_iteration_cubic(z, to_compute, c):
    #     z[to_compute] = z[to_compute]**3 + c
    #     return z

    # size = 10000
    # mpiSet = mpiJulia.MPIJulia(
    #     size, size,
    #     iteration_function=my_first_iteration_cubic
    # )

        
    # fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 200, 0.01 + 0.958j)
    # if mpiSet.rank == 0:
    #     res = np.concatenate(fractal, axis=1)
    #     renderer.save_fractal(res, "./graphics/advanced_fractal_cubic_2.jpg")

    # fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 500, 0.21 + 0.958j)
    # if mpiSet.rank == 0:
    #     res = np.concatenate(fractal, axis=1)
    #     renderer.save_fractal(res, "./graphics/advanced_fractal_cubic_3.jpg")

    # # ======================================================
    # # This one deserves more zoom!
    # # ======================================================

    # size = 15000
    # mpiSet = mpiJulia.MPIJulia(
    #     size, size,
    #     iteration_function=my_first_iteration_cubic
    # )

    # fractal = mpiSet.compute_fractal_distributed((-1, 1), (-1, 1), 500, 0.21 + 0.958j)
    # if mpiSet.rank == 0:
    #     res = np.concatenate(fractal, axis=1)
    #     renderer.save_fractal(res, "./graphics/advanced_fractal_cubic_3_zoom.jpg")

    # ======================================================
    # So, what would happen with a quintic iteration function with some twist?
    # ======================================================
    size = 2000

    def my_first_iteration_quintic(z, to_compute, c):
        z[to_compute] = z[to_compute]**5/(1 + z[to_compute]**2) + c
        return z

    mpiSet = mpiJulia.MPIJulia(
        size, size,
        iteration_function=my_first_iteration_quintic
    )

    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 500, -0.14+ 0.3j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/advanced_fractal_quintic_4.jpg")

    # ======================================================
    # Fractals and transcendental functions could definitely be friends!
    # ======================================================
    def my_first_iteration_sin(z, to_compute, c):
        z[to_compute] = np.sin(z[to_compute]) + c
        return z

    mpiSet = mpiJulia.MPIJulia(
        size, size,
        iteration_function=my_first_iteration_sin
    )

    fractal = mpiSet.compute_fractal_distributed((-0.5, 0.5), (-0.5, 0.5), 500, 0.235 + 0.2214j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/advanced_fractal_sin_5.jpg")


    def my_first_iteration_sin_cos(z, to_compute, c):
        z[to_compute] = np.sin(z[to_compute]) + np.cos(z[to_compute]) + c
        return z

    size=10000
    mpiSet = mpiJulia.MPIJulia(
        size, size,
        iteration_function=my_first_iteration_sin_cos
    )

    fractal = mpiSet.compute_fractal_distributed((-0.2, 0.2), (-0.1, 0.3), 1000, 0.6 - 0.1j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/advanced_fractal_sin_cos_6.jpg")


    # pixel_matrices = s2.interpolate_ fractal(0.28+0.008j, 0.28+0.01j, 200, (-2, 2), (-2, 2))
    
    # if s2.rank == 0:
    #     print(pixel_matrices)
    #     print(len(pixel_matrices[0]))
    #     global_fractals = []
    #     for frac_batch in pixel_matrices:
    #         num_blocks = len(frac_batch) // (size * size)
    #         blocks = [frac_batch[i * (size * size): (i + 1) * (size * size)] for i in range(num_blocks)]
    #         for block in blocks:
    #             block = np.reshape(block, (size, size))
    #             global_fractals.append(block)
    #     renderer.animate_transition(global_fractals)
    #     print(global_fractals)

    # fractal = s2.compute_fractal_distributed((-2, 2), (-2,2), 1000, grid)
    # if s2.rank == 0:
    #     res = np.concatenate(fractal, axis=1)
    #     renderer.save_fractal(res, "brot.jpg")

