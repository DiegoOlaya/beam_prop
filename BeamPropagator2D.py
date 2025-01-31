import numpy as np
import warnings

class BeamPropagator2D:
    '''An object containing the methods and results for 2+1D beam propagation.
    '''
    def __init__(self, wavelen:float, index:float=1) -> None:
        '''Create an instance of the 2D beam propagator class.

        Parameters
        ----------
        wavelen : float
            The wavelength of light to be propagated in physical units.
        index : float, optional (default = 1)
            A constant index of refraction for the medium. Can be overridden with
            class methods.

        Returns
        -------
        BeamPropagator
            The created instance of the beam propagator class.
        '''
        # Create dictionary holding flags for post-FFT transformations.
        # For example, absorbing boundary conditions, index of refraction modulation, etc.
        # Methods implementing such items should add entries to this dictionary and add a 
        # corresponding entry in the flag handler method.
        self.flags = dict()
        self.idx = index
        self.wl = wavelen
        # Define storage for the properties of each of the simulation dimensions.
        self.sim_dims = dict()

    ## -------------------------------------------- ##

    ## Universal dimension setter. ##
    def set_dimension_array(
        self,
        dim:str,
        start:float, 
        end:float,
        num_samples:int=None,
        step_size:float=None,
    ) -> bool:
        if dim not in ['x', 'y', 'z']:
            raise ValueError("The dimension must be one of 'x', 'y', or 'z'.")
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        if num_samples is not None:
            self.sim_dims[dim] = [start, end, num_samples]
            return True
        # Executes if only the step size is provided.
        num_samples = np.ceil(abs(end - start) / step_size)
        self.sim_dims[dim] = [start, end, num_samples]
        return True
    
    ## Individual dimension setters for convenience. ##
    def set_x_dimension(
        self,
        start:float, 
        end:float, 
        num_samples:int=None, 
        step_size:float=None,
    ) -> bool:
        return self.set_dimension_array('x', start, end, num_samples, step_size)
    
    def set_y_dimension(
        self,
        start:float, 
        end:float, 
        num_samples:int=None, 
        step_size:float=None,
    ) -> bool:
        return self.set_dimension_array('y', start, end, num_samples, step_size)
    
    def set_z_dimension(
        self,
        start:float, 
        end:float, 
        num_samples:int=None, 
        step_size:float=None,
    ) -> bool:
        return self.set_dimension_array('z', start, end, num_samples, step_size)
    
    ## -------------------------------------------- ##

    ## Universal dimension getter. ##
    def get_dimension_array(self, dim:str) -> np.ndarray:
        if dim not in ['x', 'y', 'z']:
            raise ValueError("The dimension must be one of 'x', 'y', or 'z'.")
        if dim not in self.sim_dims:
            raise ValueError("The dimension has not been set.")
        start, end, num_samples = self.sim_dims[dim]
        return np.linspace(start, end, num_samples)
    
    ## Individual dimension getters for convenience. ##
    def get_x_dimension(self) -> np.ndarray:
        return self.get_dimension_array('x')
    
    def get_y_dimension(self) -> np.ndarray:
        return self.get_dimension_array('y')
    
    def get_z_dimension(self) -> np.ndarray:
        return self.get_dimension_array('z')
    
    ## -------------------------------------------- ##

    ## Set the initial electric field. ##
    def set_init_Efield(self, E0:np.ndarray):
        '''Store initial electric field amplitude as an instance variable.

        Parameters
        ----------
        E0 : np.ndarray
            The amplitude values at each spatial position sampled. Must have shape
            given by (`ysamples`, `xsamples`).

        Raises
        ------
        ValueError
            If the length of the initial field does not match the dimensions specified
            by the `x` and `y` dimensions stored in the object.
        '''
        sample_area_dims = np.array([self.sim_dims['y'][2], self.sim_dims['x'][2]])
        if not np.array_equal(np.shape(E0), sample_area_dims):
            raise ValueError("Initial field array has improper dimensions.")
        self.E0 = E0