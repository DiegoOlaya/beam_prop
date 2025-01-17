import numpy as np

def aperture_mask(x, low_bound:float, high_bound:float, index = False):
    '''Produces a binary mask array defining an aperture.
    
    Parameters
    ----------
    x : list-like
        A vector of positions.
    low_bound : float
        A lower bound value for the aperture. A floating point value if comparing positions,
        otherwise an integer corresponding to an index in `x`.
    high_bound : float
        An upper bound value for the aperture. A floating point value if comparing positions,
        otherwise an integer corresponding to an index in `x`.
    index : bool, optional
        A flag determining whether the mask is generated by list index or position (default is False).

    Returns
    -------
    list-like
        A list of the same length as `x` containing either zeros or ones, representing an optical
        aperture.
    '''
    mask = np.zeros(len(x))
    if index:
        mask[low_bound:(high_bound + 1)] = 1
        return mask
    # If index=False, aperture is based on location.
    return [1 if low_bound <= pos <= high_bound else 0 for pos in x]

def gaussian_amp(A:float, x, waist:float, mu:float) -> np.ndarray:
    '''Generate array of gaussian amplitude values for the sampled x values.
    
    :param A: The amplitude of the wave.
    :type A: float 
    :param x: Array of x-positions at which to evaluate the amplitude.
    :type x: list
    :param waist: The beam waist radius of the gaussian profile.
    :type waist: float 
    :param mu: The maximum intensity position of the gaussian profile.
    :type mu: float 

    Returns
    -------
    ndarray
        A list of values corresponding to the real amplitude at the given x values.
    '''
    arg = -1*((x - mu) / waist)**2
    return A * np.exp(arg)

def lens_phase_transform(x, focal_len:float, wavelen:float) -> np.ndarray:
    '''Returns the paraxial approximation of the thin-lens phase transfer function.

    Parameters
    ----------
        x : list-like
            A vector of positions at which the lens exists.
        focal_len : float
            The focal length of the lens in physical units.
        wavelen: float
            The wavelength of the incident light in physical units.

    Returns
    -------
    np.ndarray
        The values of the phase transfer function at each point in `x`.
    '''
    return np.exp((1.0j) * (np.pi / (wavelen * focal_len)) * np.square(x))

def update_idx_intensity(
    idx_arr:np.ndarray, 
    intensity_arr:np.ndarray, 
    int_coeff:float = None,
) -> np.ndarray:
    '''Update the index of refraction at each sampling position using the
    computed intensity values.

    Parameters
    ----------
    idx_arr : np.ndarray
        Array of index of refraction for the previous time-step.
    intensity_arr : np.ndarray
        Array of computed intensities for the previous time-step.
    int_coeff : float
        The coefficient determining how much the intensity contributes to 
        the new index array.

    Returns
    -------
    np.ndarray
        The new array of index of refraction values at each sampling position.
    '''
    return idx_arr + int_coeff * intensity_arr

def update_idx_grad_I(
    idx_arr:np.ndarray, 
    intensity_arr:np.ndarray,
    spacing:float, 
    int_coeff:float = 0,
) -> np.ndarray:
    '''Update the index of refraction at each sampling position using the
    derivative of the intensity along the x-dimension.

    Parameters
    ----------
    idx_arr : np.ndarray
        Array of index of refraction for the previous time-step.
    intensity_arr : np.ndarray
        Array of computed intensities for the previous time-step.
    spacing : float
        The space between each sampling position in the x-dimension.
    int_coeff : float
        The coefficient determining how much the intensity contributes to 
        the new index array.

    Returns
    -------
    np.ndarray
        The new array of index of refraction values at each sampling position.
    '''
    # Compute gradient in x-dimension. axis=1 takes derivative across columns in numpy.
    grad_I = np.gradient(intensity_arr, spacing, axis=1)
    return idx_arr + int_coeff * grad_I