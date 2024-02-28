from mpi4py import MPI
import numpy as np

import core.FractalBase as FractalBase
import utils.Constants as Constants
import time

class MPIJulia(FractalBase.Fractal):
    """
    This object manages the distributed computation of Julia Set fractals using the 
    mpi api for python. Just like for the sequential counterpart, we have heavy use of numpy
    matrices and vectorized operations.
    """
    def __init__(self,
                points_per_row: int,
                points_per_col: int,
                iteration_function=Constants.base_julia_iteration,
                escape_time_function=Constants.base_julia_escape_time):
        # First we store the total number of points and also prepare the environment
        # for MPI execution.
        super().__init__(points_per_row, points_per_col)

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
        if (self.rank == 0):
            print(f"Mpi Communicator set up: {self.size} total nodes")

        self.iteration_function = iteration_function
        self.escape_time_function = escape_time_function


    def compute_fractal(self,
                        real_interval: tuple[np.float64, np.float64],
                        imag_interval: tuple[np.float64, np.float64],
                        max_iterations: int = 200,
                        c: np.complex128 = 0 + 0j):   

        pass

    def compute_fractal_distributed(self,
                    real_interval: tuple[np.float64, np.float64],
                    imag_interval: tuple[np.float64, np.float64],
                    max_iterations: int = 200,
                    c: np.complex128 = 0 + 0j):   

        """
        This method implements the computation of a fractal in a distributed fashion: each node has enough information to 
        find how many points need to be computed. The computation takes place, and finally  
        the result is gathered by the first process via an MPI Gather operation. 
        """

        global_pixels = None
        # The distribution occurrs by sectioning the real axis among all the nodes.
        distributed_elements = []

        n_local_elements = self.ppr
        distributed_elements_count = n_local_elements // self.size
        remainder = n_local_elements % self.size

        distributed_elements = [distributed_elements_count for i in range(self.size)]
        for i in range(remainder):
            distributed_elements[i] += 1
        
        if self.rank == 0:
            print(f"Distribution of points: {distributed_elements}")

        # Now each node builds its own computational domain. 
        n_local_elements = distributed_elements[self.rank]
        real_step = (real_interval[1] - real_interval[0])/self.ppr
        
        starting_real = real_interval[0] + real_step * sum(distributed_elements[:self.rank])
        ending_real = starting_real + real_step * distributed_elements[self.rank]

        print(f"Node {self.rank} -> ({starting_real} -> {ending_real})")

        pixels = np.zeros((self.ppc, n_local_elements), dtype=int)

        real_axis = np.linspace(starting_real, ending_real, num=n_local_elements)
        imag_axis = np.linspace(imag_interval[0], imag_interval[1], num=self.ppc)

        real, imag = np.meshgrid(real_axis, imag_axis)
        grid = real + imag * 1j
        # Then, build the boolean mask to choose which point should be computed
        to_compute = np.ones((self.ppc, n_local_elements), dtype=bool)
        diverging = np.zeros((self.ppc, n_local_elements), dtype=bool)
        
        # And apply the iteration.
        for iteration in range(max_iterations):
            grid = self.iteration_function(grid, to_compute, c)
            diverging = self.escape_time_function(grid, to_compute)

            diverging = np.reshape(diverging, (self.ppc, n_local_elements))
            pixels[diverging] = iteration
            to_compute = to_compute & np.logical_not(diverging)

            # print(f"Iteration: {iteration}")
        
        # Finally, the results are gathered in the rank 0 process.
        
        t2 = time.perf_counter()
        global_pixels = self.comm.gather(pixels, root=0)
        return global_pixels


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
        """
        This method commands the interapolation of a fractal through a span 
        of complex constants c. Parallelization happens with mpi, but it's not within
        a single fractal. Instead, the interval of c is subdivided among all the mpi processes
        that compute a single interpolation and the return it. 
        """    

        # First identify the total number of fractals each process is associated to.
        num_fractals = [steps//self.size for i in range(self.size)]
        displacements = []

        displacements.append(0)
        for i in range(self.size):
            if i < steps % self.size:
                num_fractals[i] += 1

        for i in range(1, self.size):
            displacements.append(displacements[i-1] + num_fractals[i])

        if (self.rank == 0):
            print(f"Subdivision of fractals: {num_fractals}")
        pass

        # Then, each process computes the span of complex constants that it should interpolate
        is_real_changing = starting_c.real != ending_c.real
        is_imag_changing = starting_c.imag != ending_c.imag

        delta_real = is_real_changing * (ending_c.real - starting_c.real) / steps
        delta_imag = is_imag_changing * (ending_c.imag - starting_c.imag) / steps

        local_starting_c = starting_c + delta_real * displacements[self.rank] + delta_imag * displacements[self.rank]*1j 
        local_ending_c = local_starting_c + delta_real * num_fractals[self.rank] + delta_imag * num_fractals[self.rank]*1j

        print(f"Rank {self.rank}, start: {local_starting_c}, end: {local_ending_c}")

        # Now that eac rank has its own interval, simply compute all the fractals as in the sequential version
        # Iterate now over the steps, and for each step compute and store the fractal in a list
        fractals = []
        for step in range(num_fractals[self.rank]):
            current_c = local_starting_c + ((delta_real * is_real_changing) + (delta_imag * is_imag_changing) * 1j) * step
            print(f"Rank {self.rank} Fractal {step}/{num_fractals[self.rank]}, c = {current_c}")
            fractals.append(self.compute_fractal(
                real_interval,
                imag_interval,
                max_iterations,
                current_c
            ))
        global_frac = []

        global_frac = self.comm.Gatherv(fractals, (global_frac, num_fractals), 0)
        return global_frac

    def execute_zoom(self,
                    objective_point: np.complex128, 
                    initial_radius: np.float64, 
                    objective_radius: np.float64,
                    reducing_ratio: np.float64 = 1.1, 
                    c: np.complex128 = 0 + 0j):
        pass