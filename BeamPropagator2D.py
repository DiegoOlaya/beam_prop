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
        '''General function to specify the dimensions of each axis of the 
        simulation region.

        Parameters
        ----------
        dim : str
            A single character noting what dimension to set. One of 'x', 'y', 
            or 'z'.
        start : float
            The leftmost position of the simulation region in the specified 
            dimension.
        end : float
            The rightmost position of the simulation region in the specified 
            dimension.
        num_samples : int, optional
            The number of sampled points along the axis, by default None. 
            Either this or `step_size` must be provided.
        step_size : float, optional
            The interval between sampled points along the axis, by default 
            None. Either this or `num_samples` must be provided. If both are 
            provided, this parameter will be ignored.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If the dimension is not one of 'x', 'y', or 'z'.
        ValueError
            If neither `num_samples` nor `step_size` are provided.
        '''
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
        '''Wrapper function which sets the properties of the simulation region 
        x dimension.

        Parameters
        ----------
        start : float
            The leftmost position of the simulation region in the x dimension.
        end : float
            The rightmost position of the simulation region in the x dimension.
        num_samples : int, optional
            The number of sampled points along the axis, by default None. 
            Either this or `step_size` must be provided.
        step_size : float, optional
            The interval between sampled points along the axis, by default 
            None. Either this or `num_samples` must be provided. If both are 
            provided, this parameter will be ignored.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If neither `num_samples` nor `step_size` are provided.
        '''
        return self.set_dimension_array('x', start, end, num_samples, step_size)
    
    def set_y_dimension(
        self,
        start:float, 
        end:float, 
        num_samples:int=None, 
        step_size:float=None,
    ) -> bool:
        '''Wrapper function which sets the properties of the simulation region 
        y dimension.

        Parameters
        ----------
        start : float
            The leftmost position of the simulation region in the y dimension.
        end : float
            The rightmost position of the simulation region in the y dimension.
        num_samples : int, optional
            The number of sampled points along the axis, by default None. 
            Either this or `step_size` must be provided.
        step_size : float, optional
            The interval between sampled points along the axis, by default 
            None. Either this or `num_samples` must be provided. If both are 
            provided, this parameter will be ignored.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If neither `num_samples` nor `step_size` are provided.
        '''
        return self.set_dimension_array('y', start, end, num_samples, step_size)
    
    def set_z_dimension(
        self,
        start:float, 
        end:float, 
        num_samples:int=None, 
        step_size:float=None,
    ) -> bool:
        '''Wrapper function which sets the properties of the simulation region 
        z dimension.

        Parameters
        ----------
        start : float
            The leftmost position of the simulation region in the z dimension.
        end : float
            The rightmost position of the simulation region in the z dimension.
        num_samples : int, optional
            The number of sampled points along the axis, by default None. 
            Either this or `step_size` must be provided.
        step_size : float, optional
            The interval between sampled points along the axis, by default 
            None. Either this or `num_samples` must be provided. If both are 
            provided, this parameter will be ignored.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If neither `num_samples` nor `step_size` are provided.
        '''
        return self.set_dimension_array('z', start, end, num_samples, step_size)
    
    ## -------------------------------------------- ##

    ## Universal dimension getter. ##
    def get_dimension_array(self, dim:str) -> np.ndarray:
        '''Return the array of sampled points in the specified dimension.

        Parameters
        ----------
        dim : str
            A single character in ['x', 'y', 'z'] denoting the dimension to get.

        Returns
        -------
        np.ndarray
            The array of sampled points in the specified dimension.

        Raises
        ------
        ValueError
            If the dimension is not one of 'x', 'y', or 'z'.
        ValueError
            If the dimension has not been set.
        '''
        if dim not in ['x', 'y', 'z']:
            raise ValueError("The dimension must be one of 'x', 'y', or 'z'.")
        if dim not in self.sim_dims:
            raise ValueError("The dimension has not been set.")
        start, end, num_samples = self.sim_dims[dim]
        return np.linspace(start, end, num_samples)
    
    ## Individual dimension getters for convenience. ##
    def get_x_dimension(self) -> np.ndarray:
        '''Return the array of sampled points in the x dimension.

        Returns
        -------
        np.ndarray
            The array of sampled points in the x dimension.

        Raises
        ------
        ValueError
            If the dimension has not been set.
        '''
        return self.get_dimension_array('x')
    
    def get_y_dimension(self) -> np.ndarray:
        '''Return the array of sampled points in the y dimension.

        Returns
        -------
        np.ndarray
            The array of sampled points in the y dimension.

        Raises
        ------
        ValueError
            If the dimension has not been set.
        '''
        return self.get_dimension_array('y')
    
    def get_z_dimension(self) -> np.ndarray:
        '''Return the array of sampled points in the z dimension.

        Returns
        -------
        np.ndarray
            The array of sampled points in the z dimension.

        Raises
        ------
        ValueError
            If the dimension has not been set.
        '''
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

    ## -------------------------------------------- ##

    ## Absorbing boundary conditions. ##
    def set_abs_bcs(self, width_factor:float):
        '''Define a mask implementing absorbing boundary conditions using `width_factor` as a rolloff parameter.
        Also changes the program behavior to implement absorbing boundary conditions.

        Parameters
        ----------
            width_factor : float
                A numeric parameter that defines how gradual the rolloff should be. Smaller numbers
                result in more of the plane being absorbing.
        '''
        # Check for the existence of necessary instance variables.
        try:
            self.sim_dims['x'] is not None and self.sim_dims['y'] is not None
        except:
            raise NameError("Need to define the x and y sampling positions first.")
        
        # Add absorbing boundary flag.
        self.flags['abs'] = [True, width_factor]
        return True
    
    def _gen_abs_bc_mask(self) -> np.ndarray:
        '''Return the array that multiplies the field to implement absorbing boundary conditions.

        The absorbing boundary conditions are currently implemented as the expression

        `abs_bs(n) = 1 - Exp[-(nx/2 - |n - nx/2|)/(abwidth)]`

        where `nx` is the length of the x sampling array and `abwidth` is a parameter 
        defined as `nx/width_factor`. This value is calculated at each integer value of 
        the interval `[0, nx]` and is rounded to five decimal places.

        Returns
        -------
        np.ndarray
            The absorbing boundary conditions mask array.
        '''
        
        # Generate the absorbing boundary mask.
        # Check for equal x and y dimensions.
        if self.sim_dims['x'][2] == self.sim_dims['y'][2]:
            # Get the mask parameters.
            length = self.sim_dims['x'][2]
            abwidth = length / self.flags['abs'][1]
            # Generate the array.
            abs_arr = np.around([1-np.exp(-1*(length/2 - np.abs(i - length/2)) / abwidth) for i in range(length)],5)
            # Generate a meshgrid to make a 2D absorbing mask. Using sparse arrays
            # to save memory, only increasing dimensions on return.
            ax, ay = np.meshgrid(abs_arr, abs_arr, sparse=True)
            return ax * ay
        # Handle unequal sampling dimensions.
        else:
            lx = self.sim_dims['x'][2]
            ly = self.sim_dims['y'][2]
            w_param = self.flags['abs'][1]
            abwidth_x = lx / w_param
            abwidth_y = ly / w_param
            # Generate 1D arrays for each dimension.
            abs_x = np.around([1-np.exp(-1*(lx/2 - np.abs(i - lx/2)) / abwidth_x) for i in range(lx)],5)
            abs_y = np.around([1-np.exp(-1*(ly/2 - np.abs(i - ly/2)) / abwidth_y) for i in range(ly)],5)
            # Make and return the 2D mask array.
            ax, ay = np.meshgrid(abs_x, abs_y, sparse=True)
            return ax * ay
        
    def remove_abs_bcs(self) -> bool:
        '''Update propagator behavior to stop using absorbing boundary conditions.

        Returns
        -------
        bool
            Returns True on success.
        '''
        self.flags['abs'] = [False, None]
        return True
    
    ## -------------------------------------------- ##
    
    ## Index of refraction modulation. ##
    def set_index_perturbation(self, idx_arr:np.ndarray) -> bool:
        '''Require the propagator to use a spatially varying index of refraction 
        perturbation.

        Parameters
        ----------
        idx_arr : np.ndarray
            A 3D array of the same dimensions as the simulation region. specifying 
            the size of the index perturbation at each point in the simulation 
            region. The array dimensions should be [z, y, x].

        Returns
        -------
        bool
            Returns True on success. The index perturbation is stored in the
            instance variable `flags['idx']`.

        Raises
        ------
        RuntimeError
            If the dimensions of the simulation region have not been set.
        ValueError
            If the dimensions of the index perturbation array do not match the
            dimensions of the simulation region.
        '''
        # Check that all dimensions are set.
        if len(self.sim_dims.keys()) != 3:
            raise RuntimeError("Need to define all three dimensions before setting perturbation.")
        # Check that the index array has the correct dimensions.
        area_dims = np.array([self.sim_dims['z'][2], self.sim_dims['y'][2], self.sim_dims['x'][2]])
        if not np.array_equal(np.shape(idx_arr), area_dims):
            raise ValueError("Index perturbation array has improper dimensions.")
        # Add index perturbation flag.
        self.flags['idx'] = [True, idx_arr]
        return True
    
    def remove_index_perturbation(self) -> bool:
        '''Stops the propagator from using an index of refraction perturbation 
        and deletes any exisiting perturbation from the propagator.

        Returns
        -------
        bool
            Returns True on success. The index perturbation that was set is 
            deleted from the propagator.

        Warnings
        --------
            If no index perturbation was set, a warning is issued.
        '''
        if 'idx' in self.flags:
            del self.flags['idx']
        else:
            warnings.warn("No index perturbation to remove.")
        return True
    
    ## -------------------------------------------- ##
