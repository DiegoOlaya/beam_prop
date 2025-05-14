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
    
    ## Getter/setter methods for the wavelength and index. ##
    def set_wavelength(self, wavelen:float) -> bool:
        '''Set the wavelength of light to be propagated.

        Parameters
        ----------
        wavelen : float
            The wavelength of light to be propagated in physical units.

        Returns
        -------
        bool
            Returns true on success.
        '''
        self.wl = wavelen
        return True
    
    def get_wavelength(self) -> float:
        '''Get the wavelength of light to be propagated.

        Returns
        -------
        float
            The wavelength of light to be propagated in physical units.
        '''
        return self.wl
    
    def set_base_index(self, index:float) -> bool:
        '''Set the base index of refraction for the medium.

        Parameters
        ----------
        index : float
            The index of refraction for the medium.

        Returns
        -------
        bool
            Returns true on success.
        '''
        self.idx = index
        return True
    
    def get_base_index(self) -> float:
        '''Get the base index of refraction for the medium.

        Returns
        -------
        float
            The index of refraction for the medium.
        '''
        return self.idx
    
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
    def get_dimension_array(self, dim:str, retstep = False) -> np.ndarray:
        '''Return the array of sampled points in the specified dimension.

        Parameters
        ----------
        dim : str
            A single character in ['x', 'y', 'z'] denoting the dimension to get.
        retstep : bool, optional
            If True, return the step size of the sampled points. Default is False.

        Returns
        -------
        np.ndarray
            The array of sampled points in the specified dimension.
        float, optional
            Only returned if `retstep` is true. The step size for the sampled points.

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
        return np.linspace(start, end, num_samples, retstep=retstep)
    
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
    
    def set_2D_index_perturbation(self, idx_arr:np.ndarray) -> bool:
        '''Require the propgator to use a spatially varying index of refraction
        perturbation.

        Parameters
        ----------
        idx_arr : np.ndarray
            A 2D array of the same dimensions as each XY plane of the simulation 
            region. The perturbation is assumed to be the same for all Z slices. 
            The array dimensions should be [y, x].

        Returns
        -------
        bool
            True on success. The index perturbation is stored in the
            instance variable `flags['idx']`.

        Raises
        ------
        RuntimeError
            If the X and Y dimensions of the simulation region have not been set.
        ValueError
            If the dimensions of the index perturbation array do not match the
            dimensions of the XY plane of the simulation region.

        Notes
        -----
        This method is preferable for the general index perturbation method when
        the perturbation is assumed to be the same for all Z slices. It only stores
        a 2D array instead of a 3D array, which saves memory.

        See Also
        --------
        set_index_perturbation : The general method for setting index perturbations.
        remove_index_perturbation : Remove the index perturbation from the propagator.
        '''
        # Check required dimensions are set.
        if ('x' not in self.sim_dims) or ('y' not in self.sim_dims):
            raise RuntimeError("Need to define x and y dimensions before setting 2D index perturbation.")
        # Check that the index array has the correct dimensions.
        area_dims = np.array([self.sim_dims['y'][2], self.sim_dims['x'][2]])
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

    ## Propagation function. ##
    def propagate(self) -> np.ndarray:
        '''Propagate the electric field through the simulation region with the 
        split-step FT method.

        This method is meant to be called once all aspects of the simulation 
        region have been defined. It requires the initial electric field be set.
        It also requires any absorbing boundary conditions or perturbations to 
        be set before running the function. The entire array of propagated fields 
        can be accessed in the class instance variable `field_steps`.

        Returns
        -------
        np.ndarray
            The final electric field after propagation through the simulation 
            region. Array shape is (y, x).

        Raises
        ------
        RuntimeError
            If the dimensions of the simulation region have not been set.
        RuntimeError
            If the initial electric field has not been set.

        Warnings
        --------
        RuntimeWarning
            If NaN values are introduced to the computation.
        '''
        # Check pre-requisite conditions.
        if len(self.sim_dims.keys()) != 3:
            raise RuntimeError("Need to define all three dimensions before propagating.")
        if self.E0 is None:
            raise RuntimeError("Need to define the initial electric field before propagating.")
        
        # Init storage data structures.
        reg_shape = (self.sim_dims['z'][2], self.sim_dims['y'][2], self.sim_dims['x'][2])
        self.field_steps = np.empty(reg_shape, dtype=complex)
        self.field_steps[0] = self.E0

        # Define the transfer function of free space.
        fx, fy = self._get_sampling_freqs()
        _, z_step = self.get_dimension_array('z', retstep=True)
        H = np.exp(2j * np.pi * np.emath.sqrt((self.idx/self.wl)**2 - (fx**2 + fy**2)) * (z_step/2))

        # Check for and store absorbing boundary conditions if needed.
        abs_bc = None
        if self.flags.get('abs', [False])[0] == True:
            abs_bc = self._gen_abs_bc_mask()
        
        # Begin the propagation loop.
        for i in range(self.sim_dims['z'][2] - 1):
            # Perform symmetrized FFT algorithm.
            field_ft = np.fft.fft2(self.field_steps[i])
            new_field = np.fft.ifft2(field_ft * H)
            # Apply any intermediate propagation transformations.
            new_field = self._handle_intermediate_flags(new_field, i, z_step)
            # Continue the split-step algorithm.
            new_field = np.fft.fft2(new_field)
            new_field = np.fft.ifft2(new_field * H)
            
            # Handle absorbing boundary conditions if needed.
            if abs_bc is not None:
                new_field = new_field * abs_bc

            # Check if any NaN values have been introduced to the computation.
            if np.isnan(new_field).any():
                warnings.warn(
                    "At least one NaN value appears in the E field at propagation step {val}.".format(val=i),
                    category=RuntimeWarning
                )
                
            # Store the new field in the corresponding array position.
            self.field_steps[i+1] = new_field
        
        # Return the final field.
        return self.field_steps[-1]
         
    ## Propagation Helper Functions ##
    def _handle_intermediate_flags(
        self, 
        field:np.ndarray, 
        iter_step:int, 
        z_step: float
    ) -> np.ndarray:
        '''Handle any transformations that need to applied to the field in the 
        intermediate step of the symmetrized split-step FT algorithm.

        Parameters
        ----------
        field : np.ndarray
            The field to be transformed.
        iter_step : int
            The index of the current iteration step in the propagation loop.
        z_step : float
            The step size in the z dimension.

        Returns
        -------
        np.ndarray
            The transformed field.
        '''
        if self.flags.get('idx', [False])[0] == True:
            field = field * self._get_idx_phase_transform(iter_step, z_step)
        return field
    
    def _get_idx_phase_transform(self, iter_step:int, z_step:float) -> np.ndarray:
        '''Checks the form of the index perturbation and returns the phase 
        transformation to apply to the field.

        The phase transformation is given by the expression 
        
        .. math::
            \\delta\\phi(x,y) = \\exp[ik_0z\\delta n(x,y)]

        Parameters
        ----------
        iter_step : int
            The index position corresponding to the active iteration step in the
            propagation loop.
        z_step : float
            The step size in the z dimension.

        Returns
        -------
        np.ndarray
            The phase transformation to apply to the field.
        '''
        idx_arr = self.flags['idx'][1]
        # If 3D index perturbation, use the z slice corresponding to the current 
        # iteration. Otherwise use the 2D array.
        if len(np.shape(idx_arr)) == 3:
            idx_arr = idx_arr[iter_step]
        
        # Use element-wise multiplication to apply the index perturbation.
        phase = np.exp((2j * np.pi / self.wl) * idx_arr * z_step)
        return phase

    def _get_sampling_freqs(self):
        '''Return the x and y sampling frequencies in sparse array form.

        The sampling frequency arrays are meant to be used in calculations that 
        take advantage of numpy's broadcasting feature to produce a complete 
        2D result.

        Returns
        -------
        fx, fy : tuple of np.ndarrays
            `fx` is an array of dimension (1 x nx) with the sampling frequencies 
            in the x dimension. `fy` is an array of dimension (ny x 1) with the 
            sampling frequencies in the y dimension.
        '''
        # Get sample spacing.
        _, dx = self.get_dimension_array('x', retstep=True)
        _, dy = self.get_dimension_array('y', retstep=True)
        # Get the sampling frequencies.
        fx = np.fft.fftfreq(self.sim_dims['x'][2], dx)
        fy = np.fft.fftfreq(self.sim_dims['y'][2], dy)
        return fx, np.transpose([fy])

    ## -------------------------------------------- ##
    