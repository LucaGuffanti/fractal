from typing import Any

import core.MPIJulia as mpiJulia
import core.SequentialJulia as seq
import utils.Renderer as renderer
from mpi4py import MPI

import numpy as np

if __name__ == "__main__":

    def mandelbrot_iteration(
        z: np.ndarray[int, np.complex128],
        to_compute: np.ndarray[Any, np.dtype[bool]],
        c
    ):
        z[to_compute] = (z[to_compute])**2 + c[to_compute]
        return z
    

    # ======================================================
    # The simplest fractal to be built... still beautiful!
    # ======================================================
    
    size = 2000
    mpiSet = mpiJulia.MPIJulia(
         size, size
    )
    
    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 1000, 0.24-0.59j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_fractal.jpg")

    # ======================================================
    # Why don't we try zooming in?
    # ======================================================
    
    fractal = mpiSet.compute_fractal_distributed((-0.7, 0.7), (-0.7, 0.7), 1000, 0.24-0.59j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_fractal_zoom.jpg")


    # ======================================================
    # Wow! I really like it... What happens with a real c? 
    # ======================================================
    
    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 1000, 0.9)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_fractal_real_1.jpg")

    # ======================================================
    # Hmm... If I used a smaller c?
    # ======================================================
    
    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 1000, 0.09)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_fractal_real_2.jpg")

    # ======================================================
    # Wait! I know where this is going
    # ======================================================
    

    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 1000, 0.0009)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_fractal_real_3.jpg")


    # ======================================================
    # Aha! It made a circle!
    # ======================================================
    

    # ======================================================
    # The world of Julia sets is reeeeally interesting...
    # 
    #   I want to go bigger! Hope that all the cores
    #   in LUMI will help
    # ======================================================
    
    size = 15000
    mpiSet = mpiJulia.MPIJulia(
        size, size)

    fractal = mpiSet.compute_fractal_distributed((-2, 2), (-2, 2), 500, 0.28 + 0.008j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_big_fractal_1.jpg")
    
    # ======================================================
    # It definitely took a little bit: with this sizes each
    # of the 32 cores is computing a little more than 7 
    # million numbers for 500 times!
    # Come on, let's go deeper
    # ======================================================

    fractal = mpiSet.compute_fractal_distributed((-0.5, 0), (-0.5, 0), 500, 0.28 + 0.008j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_big_fractal_1_zoom.jpg")
    
    fractal = mpiSet.compute_fractal_distributed((-1, 1), (-1, 1), 500, -0.79 + 0.15j)
    if mpiSet.rank == 0:
        res = np.concatenate(fractal, axis=1)
        renderer.save_fractal(res, "./graphics/basic_big_fractal_2.jpg")
    

    # ======================================================
    # That's really nice! Let's get to something more
    # beautiful. Take a look at advanced.py
    # ======================================================
    
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

