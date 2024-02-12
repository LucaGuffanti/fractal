from typing import Any

import numpy as np
import time

import src.core.FractalBase as FractalBase
import src.utils.Constants as Constants



class SequentialJulia(FractalBase.Fractal):
    """
    This objects manages the sequential construction of a Julia set, using vectorized numpy operations.
    Despite the loss of performance this object allows the definition of a custom iteration function which
    is applied to each point when building the set.
    """

    def __init__(self,
                 points_per_row: int,
                 points_per_col: int,
                 iteration_function=Constants.base_julia_iteration,
                 escape_time_function=Constants.base_julia_escape_time):
        # Store the total number of points.
        super().__init__(points_per_row, points_per_col)
        print(f"Points per row: {self.ppr}")
        print(f"Points per column: {self.ppc}")

        # And set up the functions
        self.iteration_function = iteration_function
        self.escape_time_function = escape_time_function

    def compute_fractal(self,
                        real_interval: tuple,
                        imag_interval: tuple,
                        max_iterations: int = 200,
                        c: np.complex128 = 0 + 0j):
        # First, build the computational domain
        pixels = np.zeros((self.ppr, self.ppc), dtype=int)

        real_axis = np.linspace(real_interval[0], real_interval[1], num=self.ppc)
        imag_axis = np.linspace(imag_interval[0], imag_interval[1], num=self.ppr)

        real, imag = np.meshgrid(real_axis, imag_axis)
        grid = real + imag * 1j
        # Then, build the boolean mask to choose which point should be computed
        to_compute = np.ones((self.ppr, self.ppc), dtype=bool)
        diverging = np.zeros((self.ppr, self.ppc), dtype=bool)

        # And apply the iteration.
        for iteration in range(max_iterations):
            grid = self.iteration_function(grid, to_compute, c)
            diverging = self.escape_time_function(grid, to_compute)

            diverging = np.reshape(diverging, (self.ppr, self.ppc))
            pixels[diverging] = iteration
            to_compute = to_compute & np.logical_not(diverging)

            # print(f"Iteration: {iteration}")
        # And return the pixels
        return pixels

    def resize_domain(self, point: np.complex128, domain_radius: np.float64):
        pass

    def interpolate_fractal(self,
                            starting_c: np.complex128,
                            ending_c: np.complex128,
                            steps: int,
                            real_interval: tuple,
                            imag_interval: tuple,
                            max_iterations: int = 200,
                            ):
        # First, understand which coordinates will change
        step_real = starting_c.real != ending_c.real
        step_imag = starting_c.imag != ending_c.imag

        if step_real:
            print("Detected change in real coordinate")
        if step_imag:
            print("Detected change in imaginary coordinate")

        # Then, compute the needed steps to move from one constant to the other.
        delta_real = (ending_c.real - starting_c.real)/steps * step_real
        delta_imag = (ending_c.imag - starting_c.imag)/steps * step_imag

        # Iterate now over the steps, and for each step compute and store the fractal in a list
        fractals = []
        for step in range(steps):
            current_c = starting_c + ((delta_real * step_real) + (delta_imag * step_imag) * 1j) * step
            print(f"Fractal {step}/{steps}, c = {current_c}")
            t1 = time.perf_counter()
            fractals.append(self.compute_fractal(
                real_interval,
                imag_interval,
                max_iterations,
                current_c
            ))
            t2 = time.perf_counter()
            print(f"    computation took {t2-t1: .5f} s")

        return fractals

    def execute_zoom(self, objective_point: np.complex128, initial_radius: np.float64, objective_radius: np.float64,
                     reducing_ratio: np.float64 = 1.1, c: np.complex128 = 0 + 0j):
        pass

    # def __init__(self,
    #              density: int,
    #              iteration_function=None):
    #     super().__init__(density, iteration_function)
    # def compute_fractal(self, cx, cy, real_interval, imag_interval, is_mandelbrot):
    #
    # def resize_domain(self,
    #                   point: complex,
    #                   domain_radius: float):
    #
    #
    #
    # def interpolate_fractal(self,
    #                         starting_c,
    #                         ending_c,
    #                         steps: int):
    #
    #     # Verify which coordinates will be changed
    #     steps_real = starting_c.real != ending_c.real
    #     steps_imag = starting_c.imag != ending_c.imag
    #     # And compute the step size
    #     delta_real = ((ending_c.real - starting_c.real) / steps) * steps_real
    #     delta_imag = ((ending_c.imag - starting_c.imag) / steps) * steps_imag
    #
    #     fractals = []
    #     c_iters = []
    #     print(f"Interpolating fractals: {starting_c}->{ending_c}...")
    #     for i in range(steps + 1):
    #         c: complex = (starting_c.real + delta_real * i) + (starting_c.imag + delta_imag * i) * 1j
    #         c_iters.append(c)
    #
    #         print(f"c : {c}, iterations : {i}/{steps}")
    #         # Interpolation of the mandelbrot set is not allowed
    #         frac = self.compute_fractal(c.real, c.imag, is_mandelbrot=False)
    #         fractals.append(frac)
    #         self.reset_structures()
    #
    #     return fractals, c_iters
    #
    # def execute_zoom(self,
    #                  objective_point: complex,
    #                  initial_radius: float,
    #                  objective_radius: float,
    #                  c):
    #     assert (initial_radius != 0 and objective_radius != 0 and initial_radius > objective_radius)
    #
    #     steps = np.ceil(np.log2(initial_radius / objective_radius)).astype(int)
    #
    #     print(f"Executing Zoom: {steps} will be required")
    #     current_radius = initial_radius
    #     fractals = []
    #     for i in range(steps + 1):
    #         print(f"radius: {current_radius}, iterations: {i}/{steps}")
    #         self.resize_domain(objective_point, current_radius)
    #
    #         frac = self.compute_fractal_user_function(self.initial_domain)
    #         renderer.render_fractal(frac)
    #         fractals.append(frac)
    #         current_radius = current_radius / 2
    #
    #     return fractals
