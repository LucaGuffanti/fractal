from typing import Any

import core.MPIJulia as mpiJulia
import utils.Renderer as renderer
from mpi4py import MPI

import numpy as np

if __name__ == "__main__":
    # Position: -0.7746806106269039 - 0.1374168856037867
    # Range: 1.506043553756164E-12
    # real_domain: tuple = (-0.7746806106269039-1.506043553756164e-12, -0.7746806106269039+1.506043553756164e-12)
    # imag_domain: tuple = (-0.1374168856037867-1.506043553756164e-12, -0.1374168856037867+1.506043553756164e-12)

    # real_domain: tuple = (-2, 2)
    # imag_domain: tuple = (-2, 2)
    # max_iterations: int = 500

    # def mandelbrot_iteration(
    #     z: np.ndarray[int, np.complex128],
    #     to_compute: np.ndarray[Any, np.dtype[bool]],
    #     c
    # ):
    #     z[to_compute] = np.abs(z[to_compute].real) + np.abs(z[to_compute].imag)*1j
    #     z[to_compute] = (z[to_compute])**2 + c[to_compute]
    #     return z


    # s = seq.SequentialJulia(grid_dim, grid_dim, mandelbrot_iteration)



    # c_start: complex = -0.835 + 0 * 1j
    # c_end: complex = -0.835 - 2 * 1j



    # c_start = -0.29609091 + 0.62491*1j
    # c_end = - 0.20509091 + 0.71591*1j

    # c_start = -0.52 - 0.5672935j
    # c_end =-0.892 - 0.5672935j

    # print("Computing fractal")

    #RE = [-0.96, -0.80]
    #IM = [-0.35, -0.2]

    # real_axis = np.linspace(-2.5, 1, grid_dim)
    # imag_axis = np.linspace(-1, 1, grid_dim)
    # real, imag = np.meshgrid(real_axis, imag_axis)
    # grid = real + imag * 1j

    # fractal = s.compute_fractal((-2.5, 1), (-1, 1), max_iterations, grid)
    # print(fractal)
    # print("Rendering fractal")
    # renderer.render_fractal(fractal)

    # print("Computing fractal")
    # fractal = s.compute_fractal((-1.3, 1.3), (-1.3, 1.3), max_iterations, c_end)
    # print(fractal)
    # print("Rendering fractal")
    # renderer.render_fractal(fractal)

    # fractals = s.interpolate_fractal(c_start, c_end,300,(-1.3, 1.3), (-1.3, 1.3), max_iterations)
    # renderer.animate_transition(fractals)



    # -0.58 -0.55i
    # fractals = s.execute_zoom(
    #     -0.58 - 0.55j,
    #     2,
    #     1 / (2 ** 30),
    #     s.initial_domain
    # )
    # renderer.animate_transition(fractals)

    
    grid_dim = 1001
    size = 500
    s2 = mpiJulia.MPIJulia(
        size, size
    )

    # pixel_matrices = s2.compute_fractal_distributed((-2, 2), (-2,2), 200, -0.85-0.2j)
    # pixel_matrices = s2.interpolate_fractal(0+0j, -0.85-0.2j, 12, (-2, 2), (-2, 2))
    # if s2.rank == 0:
    #     # print(pixel_matrices)
    #     # print(len(pixel_matrices))
    #     # print(len(pixel_matrices[0]))
    #     res = np.concatenate(pixel_matrices, axis=1)
    #     print(len(res))
    #     renderer.save_fractal(res, "frac.jpg")

    pixel_matrices = s2.interpolate_fractal(0+0j, -0.85-0.2j, 200, (-2, 2), (-2, 2))
    
    if s2.rank == 0:
        print(pixel_matrices)
        print(len(pixel_matrices[0]))
        global_fractals = []
        for frac_batch in pixel_matrices:
            num_blocks = len(frac_batch) // (size * size)
            blocks = [frac_batch[i * (size * size): (i + 1) * (size * size)] for i in range(num_blocks)]
            for block in blocks:
                block = np.reshape(block, (size, size))
                global_fractals.append(block)
        renderer.animate_transition(global_fractals)
        print(global_fractals)

