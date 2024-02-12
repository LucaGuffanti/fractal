from abc import ABC, abstractmethod

import numpy as np


class Fractal(ABC):
    """
    Abstract class that defines methods used to compute a fractal via the iterative application of a formula
    The base fractal class is inherited by both sequential and parallel implementations of the mandelbrot and julia
    set builders.
    """

    def __init__(self,
                 points_per_row: int = 1000,
                 points_per_col: int = 1000):
        """
        Builds the objects, storing the total number of points that constitute the computational domain,
        which is stored as a grid.
        :param points_per_row: number of points per row of the grid
        :param points_per_col: number of points per columns of the grid
        """
        self.ppr = points_per_row
        self.ppc = points_per_col

    @abstractmethod
    def compute_fractal(self,
                        real_interval: tuple[np.float64, np.float64],
                        imag_interval: tuple[np.float64, np.float64],
                        max_iterations: int = 200,
                        c: np.complex128 = 0 + 0j):
        """
        Applies the iteration formula.
        :param real_interval: span of real numbers
        :param imag_interval: span of imaginary numbers
        :param c: complex constant used for the iteration
        :param max_iterations: maximum number of iterations allowed per point

        :return: a numpy matrix containing the number of iterations reached by each point
        """
        pass

    @abstractmethod
    def resize_domain(self,
                      point: np.complex128,
                      domain_radius: np.float64):
        """
        Resizes the computational domain
        :param point: point around which the domain is resized
        :param domain_radius: radius of the new domain
        """
        pass

    @abstractmethod
    def interpolate_fractal(self,
                            starting_c: np.complex128,
                            ending_c: np.complex128,
                            steps: int,
                            real_interval: tuple,
                            imag_interval: tuple,
                            max_iterations: int = 200,
                            ):
        """
        Interpolates all the fractals that are built as the complex constant varies

        :param starting_c: starting complex constant
        :param ending_c: objective complex constant
        :param steps: total number of steps to be executed
        :return: a list of numpy matrices, each one describing a single fractal
        :param real_interval: interval of real numbers for which the fractal is computed
        :param imag_interval: interval of complex numbers for which the fractal is computed
        :param max_iterations: maximum number of iterations allowed per fractal

        """
        pass

    @abstractmethod
    def execute_zoom(self,
                     objective_point: np.complex128,
                     initial_radius: np.float64,
                     objective_radius: np.float64,
                     reducing_ratio: np.float64 = 1.1,
                     c: np.complex128 = 0 + 0j):
        """
        Produces a zoom by iteratively computing new fractals
        :param objective_point: Point around which the zoom is performed
        :param initial_radius: initial radius of the computation domain
        :param objective_radius: objective radius of the computational domain
        :param reducing_ratio: ratio at which the intervals are scaled down
        :param c: complex constant defining the fractal, defaulted to 0.
        :return: a list of numpy matrices, each one describing a single fractal
        """
        pass
