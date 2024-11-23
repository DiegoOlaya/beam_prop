import numpy as np
import warnings

class BeamPropagator:
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

    ### Helper methods ###
    def _handle_post_flags(self, E_field:np.ndarray) -> np.ndarray:
        '''Handles any post FT transformations done on the E-field. What transformations
        to perform are provided by the `flags` instance variable, a dictionary of boolean values
        naming each transformation.

        Parameters
        ----------
        E_field : np.ndarray
            The complex field amplitude before any transformations are applied.

        Returns
        -------
        np.ndarray
            The complex field amplitude after applying all defined transformations.
        '''
        if self.flags.get('abs') == True:
            E_field = E_field * self.abs_arr
        return E_field
            
    def _handle_int_flags(self, E_field:np.ndarray) -> np.ndarray:
        '''Handles intermediate transformations done on the E-field during the symmetrized split-step
        operation. Transformation are provided by the `flags` instance variable, a dictionary of boolean values
        naming each transformation.

        Parameters
        ----------
        E_field : np.ndarray
            The complex field amplitude before any transformations are applied.

        Returns
        -------
        np.ndarray
            The complex field amplitude after applying all defined transformations.
        '''
        if self.flags.get('idx_pert') == True:
            E_field = E_field * self.idx_pert
        return E_field

    ### Class methods. ###
    def set_base_index(self, new_n:float):
        '''Change the base index of refraction of the propagator class.

        Parameters
        ---------
        new_n : float
            The new constant index of refraction for the propagation region.
        '''
        self.idx = new_n

    def set_x_array(
        self, 
        x_length:float, 
        num_samples:int=None, 
        step_size:float=None,
        x_array:np.ndarray = None,
    ) -> np.ndarray:
        '''Defines the array of x values considered by the propagator. Must provide one of `num_samples` or
        `step_size`. If both are provided, `num_samples` takes priority. Sets the `x_arr` and `x_step` class 
        variables and returns the x-value array. Alternatively, one can provide an array to be the x-sampling
        array for the `BeamPropagator` object to store. Providing this array will override all other function
        behavior.

        Parameters
        ----------
        x_length : float
            The length in physical units of the system.
        num_samples : int, optional
            An integer number representing the number of samples along that dimension.
        step_size : float, optional
            The step size in physical units between sample points.
        x_array : np.ndarray, optional
            The array of values corresponding to the desired x-coordinate sampling locations.

        Raises
        ------
        ValueError
            If neither the number of samples or the step size is provided.

        Returns
        -------
        np.ndarray
            An array containing evenly spaced points in [`-x_length/2`, `x_length/2`].
        '''
        # Check if array parameter is passed.
        if x_array is not None:
            self.x_arr = x_array
            self.x_step = np.abs(x_array[1] - x_array[0])
            return self.x_arr
        # Check if we have enough information to set array.
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        # Number of samples will override the step size behavior.
        if num_samples is not None:
            samples, step = np.linspace(start= -1 * x_length / 2, stop = x_length / 2, num = num_samples, retstep=True)
            # Store the resulting array and step into class variables.
            self.x_arr = samples
            self.x_step = step
            # Returns the x_array for computation.
            return samples
        # Executes if only the step size is provided.
        num_samples = np.ceil(x_length / step_size)
        samples, step = np.linspace(start= -1 * x_length / 2, stop = x_length / 2, num = num_samples, retstep=True)
        self.x_arr = samples
        self.x_step = step
        return samples

    def set_z_array(
        self, 
        z_length:float, 
        num_samples:int=None, 
        step_size:float=None,
        z_offset:float=0.0,
    ) -> np.ndarray:
        '''Defines the array of z values considered by the propagator. Must provide one of `num_samples` or
        `step_size`. If both are provided, `num_samples` takes priority. Sets the `z_arr` and `z_step` class 
        variables and returns the z-value array.

        Parameters
        ----------
        z_length : float
            The length in physical units of the system.
        num_samples : int, optional
            An integer number representing the number of samples along that dimension.
        step_size : float, optional
            The step size in physical units between sample points.
        z_offset : float, optional (default=0.0)
            An alternate value for the zero coordinate that offsets the array accoringly.

        Raises
        ------
        ValueError
            If neither the number of samples or the step size is provided.

        Returns
        -------
        np.ndarray
            An array containing evenly spaced points in [`0`, `z_length`] + z_offset (if defined).
        '''
        # Check if we have enough information to set array.
        if (num_samples is None and step_size is None):
            raise ValueError("Either the number of samples or the step size must be provided.")
        # Number of samples will override the step size parameter.
        if num_samples is not None:
            samples, step = np.linspace(start = 0, stop = z_length, num = num_samples, retstep=True)
            # Store the resulting array and step into class variables.
            self.z_arr = samples
            self.z_step = step
            # Returns the z_array for computation.
            return samples
        # Executes if only the step size is provided.
        num_samples = np.ceil(x_length / step_size)
        samples, step = np.linspace(start = 0, stop = z_length, num = num_samples, retstep=True)
        self.z_arr = samples + z_offset
        self.z_step = step
        return samples

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

    def set_abs_bcs(self, width_factor:float):
        '''Define a mask implementing absorbing boundary conditions using `width_factor` as a rolloff parameter.
        Also changes the program behavior to implement absorbing boundary conditions.

        The absorbing boundary conditions are currently implemented as the expression

        `abs_bs(n) = 1 - Exp[-(nx/2 - |n - nx/2|)/(abwidth)]`

        where `nx` is the length of the x sampling array and `abwidth` is a parameter defined as `nx/width_factor`.
        This value is calculated at each integer value of the interval `[0, nx]` and is rounded to five decimal places.

        Parameters
        ----------
            width_factor : float
                A numeric parameter that defines how gradual the rolloff should be. Smaller numbers
                result in more of the plane being absorbing.
        '''
        # Check for the existence of necessary instance variables.
        try:
            self.x_arr is not None
        except:
            raise NameError("Need to define the x sampling positions first.")
        
        # Add absorbing boundary flag.
        self.flags['abs'] = True
        # Define intermediate parameters.
        nx = len(self.x_arr)
        abwidth = nx / width_factor
        # Set absorbing boundary condition instance variable.
        self.abs_arr = np.around([1-np.exp(-1*(nx/2 - np.abs(i - nx/2)) / abwidth) for i in range(nx)],5)

    def remove_abs_bcs(self):
        '''Update propagator behavior to stop using absorbing boundary conditions.
        '''
        self.flags['abs'] = False

    def set_x_idx_pertubation(self, inhom_form:np.ndarray):
        '''Tells the propagator to include an inhomogeneous index in the propgation region defined by
        the user. Assumes a small perturbation of the index of refraction, where the index is of the form
        `n + dn`.

        Parameters
        ----------
        inhom_form : np.ndarray
            The values of the user-defined index fluctuation at all points in the plane 
            of the wave. Must be the same fluctuation over the whole propagation region.
        '''
        try:
            self.z_step is not None
        except:
            raise NameError("Need to define the z sampling positions first.")
        # Set flag.
        self.flags['idx_pert'] = True
        # Set index perturbation.
        self.idx_pert = np.exp(2j * np.pi * self.z_step * inhom_form / self.wl)

    def rm_x_idx_perturbation(self):
        '''Turns off behavior accounting for a perurbation in the index of refraction.
        '''
        self.flags['idx_pert'] = False
    
    def propagate(self) -> np.ndarray:
        '''Uses the split-step Fourier transform method to compute the complex field amplitude
        at each z position specified by the object's `z` array.

        Returns
        -------
        np.ndarray
            The complex field amplitude at the final step of propagation.
        '''
        # Create empty array to store the intermediate field configurations.
        self.field_steps = np.empty((len(self.z_arr), len(self.E0)),dtype=complex)
        # Initialize the first element with the given initial field configuration.
        self.field_steps[0] = self.E0
        # Define the free-space propagation trasnfer function in Fourier space.
        freqs = self.get_fx_values()
        H = np.exp(-2j * np.pi * np.emath.sqrt((self.idx / self.wl)**2 - (freqs)**2) * (self.z_step/2))
        # Perform the split-step algorithm for each element in the array.
        for i in range(len(self.field_steps)-1):
            # Performs symmetrized split-step algorithm.
            field_ft = np.fft.fft(self.field_steps[i])
            new_field = np.fft.ifft(field_ft * H)
            new_field = self._handle_int_flags(new_field)
            new_field = np.fft.fft(new_field)
            new_field = np.fft.ifft(new_field * H)
            # Handle post-FT transformations.
            new_field = self._handle_post_flags(new_field)
            # Check if any NaN values have been introduced to the computation.
            if np.isnan(new_field).any():
                warnings.warn("At least one NaN value appears in the E field at propagation step {val}.".format(val=i),
                              category=RuntimeWarning)
            # Store the result of the step in the next array position.
            self.field_steps[i+1] = new_field
        # Return the final field configuration after propagation.
        return self.field_steps[-1]
    
    def get_fx_values(self):
        '''Returns the array corresponding to the frequency components of the 
        np.fft.fft function, using the given field parameters. This array corresponds
        to the transverse frequency components as determined by the x-dimension sampling array.
        '''
        return np.fft.fftfreq(self.E0.size, self.x_step)    