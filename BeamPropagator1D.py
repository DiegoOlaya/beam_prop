import numpy as np
import warnings

class BeamPropagator1D:
    '''An object containing the methods and results for 1+1D beam propagation.
    '''

    def __init__(self, wavelen:float, index:float=1) -> None:
        '''Create an instance of the beam propagator class.

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
            A single character noting what dimension to set. One of 'x' or 'z'.
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
            If the dimension is not one of 'x' or 'z'.
        ValueError
            If neither `num_samples` nor `step_size` are provided.
        '''
        if dim not in ['x', 'z']:
            raise ValueError("The dimension must be one of 'x' or 'z'.")
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        if num_samples is not None:
            self.sim_dims[dim] = [start, end, num_samples]
            return True
        # Executes if only the step size is provided.
        num_samples = np.ceil(abs(end - start) / step_size)
        self.sim_dims[dim] = [start, end, num_samples]
        return True
    
    def set_dimension_from_array(
        self,
        dim:str,
        arr:np.ndarray,
    ) -> bool:
        '''Set a dimension of the simulation region from a provided array, 
        taken to represent the sampled points along that dimension.

        Parameters
        ----------
        dim : str
            A single character noting what dimension to set. One of 'x' or 'z'.
        arr : np.ndarray
            An array of values corresponding to the desired sampling locations.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties
            are stored in the class instance variable `sim_dims` as a list of
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If the dimension is not one of 'x' or 'z'.
        '''
        if dim not in ['x', 'z']:
            raise ValueError("The dimension must be one of 'x' or 'z'.")
        self.sim_dims[dim] = [arr[0], arr[-1], len(arr)]
        return True
    
    ## Individual dimension setters for convenience. ##
    def set_x_dimension(
        self, 
        start:float,
        end:float,
        num_samples:int = None, 
        step_size:float = None,
    ) -> bool:
        '''Defines the array of x values considered by the propagator. Must 
        provide one of `num_samples` or `step_size`. If both are provided, 
        `num_samples` takes priority.

        Parameters
        ----------
        start : float
            The leftmost position of the simulation region in the specified 
            dimension.
        end : float
            The rightmost position of the simulation region in the specified 
            dimension.
        num_samples : int, optional
            An integer number representing the number of samples along that 
            dimension.
        step_size : float, optional
            The step size in physical units between sample points.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If neither the number of samples or the step size is provided.
        '''
        # Check if we have enough information to set array.
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        
        return self.set_dimension_array(
            dim = 'x',
            start = start,
            end = end,
            num_samples = num_samples,
            step_size = step_size
        )

    def set_z_dimension(
        self, 
        end:float,
        start:float = 0.0, 
        num_samples:int=None, 
        step_size:float=None,
        z_offset:float=0.0,
    ) -> bool:
        '''Defines the array of z values considered by the propagator. Must 
        provide one of `num_samples` or `step_size`. If both are provided, 
        `num_samples` takes priority. Sets the `z_arr` and `z_step` class 
        variables and returns the z-value array.

        Parameters
        ----------
        end : float
            The length in physical units of the system.
        start : float, optional (default=0.0)
            The starting position of the simulation region in the z-dimension.
        num_samples : int, optional
            An integer number representing the number of samples along that dimension.
        step_size : float, optional
            The step size in physical units between sample points.
        z_offset : float, optional (default=0.0)
            An alternate value for the zero coordinate that offsets the array 
            accoringly.

        Returns
        -------
        bool
            Returns true on successful execution. The dimension's properties 
            are stored in the class instance variable `sim_dims` as a list of 
            form [start, end, num_samples].

        Raises
        ------
        ValueError
            If neither the number of samples or the step size is provided.
        '''
        # Check if we have enough information to set array.
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        # Handle offset if provided.
        if z_offset != 0.0:
            start += z_offset
            end += z_offset
        
        return self.set_dimension_array(
            dim = 'z', 
            start = start, 
            end = end,
            num_samples = num_samples, 
            step_size = step_size
        )

    ## -------------------------------------------- ##

    ## Universal dimension getter. ##
    def get_dimension_array(self, dim:str, retstep:bool=False) -> np.ndarray:
        '''Return the array of sampled points in the specified dimension.

        Parameters
        ----------
        dim : str
            A single character in ['x', 'z'] denoting the dimension to get.
        retstep : bool, optional
            If True, return the step size of the sampled points. Default is 
            False.

        Returns
        -------
        np.ndarray
            The array of sampled points in the specified dimension.
        float, optional
            Only returned if `retstep` is true. The step size for the sampled 
            points.

        Raises
        ------
        ValueError
            If the dimension is not one of 'x' or 'z'.
        ValueError
            If the dimension has not been set.
        '''
        if dim not in ['x', 'z']:
            raise ValueError("The dimension must be one of 'x' or 'z'.")
        if dim not in self.sim_dims:
            raise ValueError("The dimension has not been set.")
        start, end, num_samples = self.sim_dims[dim]
        return np.linspace(start, end, num_samples, retstep=retstep)
    
    ## Dimension specific getters. ##
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

    ## Set initial electric field. ##
    def set_init_Efield(self, E0:np.ndarray):
        '''Store initial electric field amplitude as a class variable.

        Parameters
        ----------
        E0 : np.ndarray
            The amplitude values at each spatial position sampled.

        Raises
        ------
        ValueError
            If the length of the initial field does not match the `x` array class variable.
        '''
        if len(E0) != len(self.x_arr):
            raise ValueError("Initial field array must have same dimensions as x dimension sample array.")
        self.E0 = E0

    ## -------------------------------------------- ##

    ## Absorbing boundary conditions. ##
    def set_abs_bcs(self, width_factor:float):
        '''Define a mask implementing absorbing boundary conditions using 
        `width_factor` as a rolloff parameter. Also changes the program 
        behavior to implement absorbing boundary conditions.

        The absorbing boundary conditions are currently implemented as the 
        expression

        .. math::
            abs_bs(n) = 1 - \\exp[-\\frac{n_x/2 - |n - n_x/2|}{abwidth}]

        where `nx` is the length of the x sampling array and `abwidth` is a 
        parameter defined as `nx/width_factor`. This value is calculated at 
        each integer value of the interval `[0, nx]` and is rounded to five 
        decimal places.

        Parameters
        ----------
            width_factor : float
                A numeric parameter that defines how gradual the rolloff should
                be. Smaller numbers result in more of the plane being 
                absorbing.
        '''
        # Check for the existence of necessary instance variables.
        try:
            self.sim_dims['x'] is not None
        except:
            raise NameError("Need to define the x sampling positions first.")
        
        # Define intermediate parameters.
        nx = self.sim_dims['x'][2]  # Number of samples in x dimension.
        abwidth = nx / width_factor
        # Set absorbing boundary condition instance variable.
        abs_arr = np.around(
            [1-np.exp(-1*(nx/2 - np.abs(i - nx/2)) / abwidth) for i in range(nx)],
            5
        )
        self.flags['abs'] = [True, abs_arr]

    def remove_abs_bcs(self):
        '''Update propagator behavior to stop using absorbing boundary 
        conditions.
        '''
        self.flags['abs'] = [False, None]

    ## -------------------------------------------- ##

    ## Index of refraction modulation. ##
    def set_index_perturbation(self, delta_idx:np.ndarray):
        '''Require the propagator to use a spatially varying index of refraction 
        perturbation. The index perturbation is stored in the instance variable
        `flags['idx']`.

        Parameters
        ----------
        delta_idx : np.ndarray
            A 2D array of the same dimensions as the simulation region. specifying
            the size of the index perturbation at each point in the simulation
            region. The array dimensions should be [z, x]. 

        Raises
        ------
        RuntimeError
            If the dimensions of the simulation region have not been set.
        ValueError
            If the dimensions of the index perturbation array do not match the
            dimensions of the simulation region.
        '''
        # Check required dimensions set.
        if len(self.sim_dims.keys()) != 2:
            raise RuntimeError("Need to define both x and z dimensions first.")
        # Check if the array has the correct shape.
        dims = (self.sim_dims['z'][2], self.sim_dims['x'][2])
        if not np.array_equal(delta_idx.shape, dims):
            raise ValueError('Dimensions do not match the sampling parameters stored.')
        # Store the perturbation in the flags dictionary.
        self.flags['idx'] = [True, delta_idx]

    def set_1D_idx_pertubation(self, delta_idx:np.ndarray):
        '''Require the propgator to use a spatially varying index of refraction
        perturbation. Assumes a small perturbation in the index of refraction 
        taken to be constant in z, but varying in x.

        Parameters
        ----------
        delta_idx : np.ndarray
            The values of the user-defined index fluctuation at all points in 
            the plane of the wave. Must be the same fluctuation over the whole 
            propagation region.

        Raises
        ------
        RuntimeError
            If the X dimension of the simulation region has not been set.
        ValueError
            If the length of the perturbation array does not match the 
            simulation X array.

        Notes
        -----
        This method is preferable for the general index perturbation method when
        the perturbation is assumed to be the same for all Z slices. It only stores
        a 1D array instead of a 2D array, which saves memory.

        See Also
        --------
        set_index_perturbation : The general method for setting index perturbations.
        remove_index_perturbation : Remove the index perturbation from the propagator.
        '''
        if 'x' not in self.sim_dims:
            raise RuntimeError("Need to define the x dimension first.")
        # Check correct array dimensions.
        if len(delta_idx) != self.sim_dims['x'][2]:
            raise ValueError("Index perturbation array must have same dimensions as simulation X array.")
        # Set flag.
        self.flags['idx'] = [True, delta_idx]

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
    
    def propagate(self) -> np.ndarray:
        '''Uses the split-step Fourier transform method to compute the complex 
        field amplitude at each z position specified by the object's `z` array.

        Returns
        -------
        np.ndarray
            The complex field amplitude at the final step of propagation.
        '''
        # Check prereqeuisites for propagation.
        if len(self.sim_dims.keys()) != 2:
            raise RuntimeError("Need to define both x and z dimensions first.")
        if self.E0 is None:
            raise RuntimeError("Need to set the initial electric field before propagation.")
        
        # Init storage data structures.
        reg_shape = (self.sim_dims['z'][2], self.sim_dims['x'][2])
        self.field_steps = np.empty(reg_shape, dtype=complex)
        self.field_steps[0] = self.E0

        # Define the free-space propagation trasnfer function in Fourier space.
        freqs = self._get_sampling_freqs()
        _, z_step = self.get_dimension_array('z', retstep=True)
        H = np.exp(2j * np.pi * np.emath.sqrt((self.idx / self.wl)**2 - (freqs)**2) * (z_step/2))

        # Perform the split-step algorithm for each element in the array.
        for i in range(self.sim_dims['z'][2] - 1):
            # Performs symmetrized split-step algorithm.
            field_ft = np.fft.fft(self.field_steps[i])
            new_field = np.fft.ifft(field_ft * H)
            new_field = self._handle_intermediate_flags(new_field, i, z_step)
            new_field = np.fft.fft(new_field)
            new_field = np.fft.ifft(new_field * H)
            
            # Absorbing boundary.
            if self.flags.get('abs', [False])[0] == True:
                new_field = new_field * self.flags['abs'][1]
            
            # Check if any NaN values have been introduced to the computation.
            if np.isnan(new_field).any():
                warnings.warn(
                    f"At least one NaN value appears in the E field at propagation step {i}.",
                    category=RuntimeWarning
                )
            
            # Store the result of the step in the next array position.
            self.field_steps[i+1] = new_field
        
        # Return the final field configuration after propagation.
        return self.field_steps[-1]

    ## Propagation Helper Methods ##
    def _get_sampling_freqs(self):
        '''Returns the array corresponding to the frequency components of the 
        np.fft.fft function, using the given field parameters. This array 
        corresponds to the transverse frequency components as determined by 
        the x-dimension sampling array.
        '''
        _, dx = self.get_dimension_array('x', retstep=True)
        return np.fft.fftfreq(self.sim_dims['x'][2], dx)
            
    def _handle_intermediate_flags(
        self, 
        field:np.ndarray, 
        iter_step:int,
        z_step:float,
    ) -> np.ndarray:
        '''Handles intermediate transformations done on the E-field during the 
        symmetrized split-step operation. Transformations are provided by the 
        `flags` instance variable, a dictionary with entries for each
        transformation.

        Parameters
        ----------
        field : np.ndarray
            The complex field amplitude before any transformations are applied.
        iter_step : int
            The step number that is currently being computed, used to index into 
            the correct row of any full-region storage structures.
        z_step : float
            The step size in the z-dimension.

        Returns
        -------
        np.ndarray
            The complex field amplitude after applying all defined transformations.
        '''
        if self.flags.get('idx', [False])[0] == True:
            field = field * self._get_idx_phase_transform(iter_step, z_step)
        return field
    
    def _get_idx_phase_transform(
        self, 
        iter_step:int, 
        z_step:float
    ) -> np.ndarray:
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
        # If 2D index perturbation, use the z slice corresponding to the current 
        # iteration. Otherwise use the 1D array.
        if len(np.shape(idx_arr)) == 2:
            idx_arr = idx_arr[iter_step]
        
        # Use element-wise multiplication to apply the index perturbation.
        phase = np.exp((2j * np.pi / self.wl) * idx_arr * z_step)
        return phase
