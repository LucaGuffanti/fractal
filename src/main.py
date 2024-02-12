from typing import Any

import src.core.julia.SequentialJulia as seq
import src.utils.Renderer as renderer
import src.utils.Constants as Constants

import numpy as np

if __name__ == "__main__":
    # Position: -0.7746806106269039 - 0.1374168856037867
    # Range: 1.506043553756164E-12
    # real_domain: tuple = (-0.7746806106269039-1.506043553756164e-12, -0.7746806106269039+1.506043553756164e-12)
    # imag_domain: tuple = (-0.1374168856037867-1.506043553756164e-12, -0.1374168856037867+1.506043553756164e-12)

    real_domain: tuple = (-2, 2)
    imag_domain: tuple = (-2, 2)
    max_iterations: int = 100

    def quintic_iteration(
        z: np.ndarray[int, np.complex128],
        to_compute: np.ndarray[Any, np.dtype[bool]],
        c: np.complex128
    ):
        z[to_compute] = z[to_compute]**5 + c
        return z

    s = seq.SequentialJulia(2000,
                            2000,
                            iteration_function=quintic_iteration)



    # c_start: complex = -0.835 + 0 * 1j
    # c_end: complex = -0.835 - 2 * 1j



    # c_start = -0.29609091 + 0.62491*1j
    # c_end = - 0.20509091 + 0.71591*1j

    c_start = -0.52 - 0.5672935j
    c_end =-0.892 - 0.5672935j

    print("Computing fractal")
    fractal = s.compute_fractal((-2.0, 2.0), (-2.0, 2.0), max_iterations, c_start)
    print(fractal)
    print("Rendering fractal")
    renderer.render_fractal(fractal)

    print("Computing fractal")
    fractal = s.compute_fractal((-1.3, 1.3), (-1.3, 1.3), max_iterations, c_end)
    print(fractal)
    print("Rendering fractal")
    renderer.render_fractal(fractal)

    fractals = s.interpolate_fractal(c_start, c_end,300,(-1.3, 1.3), (-1.3, 1.3), max_iterations)
    renderer.animate_transition(fractals)



    # -0.58 -0.55i
    # fractals = s.execute_zoom(
    #     -0.58 - 0.55j,
    #     2,
    #     1 / (2 ** 30),
    #     s.initial_domain
    # )
    # renderer.animate_transition(fractals)
