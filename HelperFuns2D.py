import numpy as np

## -------------------------------------------- ##

## Apertures ##
def rect_aperture(
    x:np.ndarray,
    y:np.ndarray,
    width_x:float,
    width_y:float,
    center:tuple,
    index=False
) -> np.ndarray:
    '''Generates a recangular aperture from the given parameters, with the 
    dimensions of the mask matching the given X and Y dimension.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the grid points.
    y : np.ndarray
        The y-coordinates of the grid points.
    width_x : float
        The width of the aperture in the x-direction.
    width_y : float
        The width of the aperture in the y-direction.
    center : tuple
        The (x, y) coordinates of the center of the rectangular aperture.
    index : bool, optional
        Tells the function whether the provided widths and `center` are
        in index units or physical units. By default False, meaning they are in
        physical units.

    Returns
    -------
    np.ndarray
        The rectangular aperture mask.
    '''
    # Initialize data structures.
    cx, cy = center
    mask_x = np.zeros_like(x)
    mask_y = np.zeros_like(y)
    # If index is True.
    if index:
        # Use integer division to get the indices that should be transparent.
        mask_x[cx - width_x//2:cx + width_x//2 + 1] = 1
        mask_y[cy - width_y//2:cy + width_y//2 + 1] = 1
        # Use meshgrid to create the 2D array representing the aperture.
        mask_x, mask_y = np.meshgrid(mask_x, mask_y, sparse=True)
        return mask_x * mask_y
    # If index is False.
    mask_x = np.where((x >= cx - width_x/2) & (x <= cx + width_x/2), 1, 0)
    mask_y = np.where((y >= cy - width_y/2) & (y <= cy + width_y/2), 1, 0)
    mask_x, mask_y = np.meshgrid(mask_x, mask_y, sparse=True)
    return mask_x * mask_y

def circ_aperture(
    x:np.ndarray,
    y:np.ndarray,
    radius:float,
    center:tuple,
    index=False
):
    pass

def sq_aperture(
    x:np.ndarray,
    y:np.ndarray,
    side_length:float,
    center:tuple,
    index=False
) -> np.ndarray:
    '''Wrapper function of `rect_aperture` for square apertures.
    Produces a square aperture mask with the given side length and center.
    The aperture is defined as a square region centered at `center` with the 
    specified `side_length`.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the grid points.
    y : np.ndarray
        The y-coordinates of the grid points.
    side_length : float
        The length of the sides of the square aperture, either in physical 
        units or in indices.
    center : tuple
        The (x, y) coordinates of the center of the square aperture, either in 
        physical units or in indices.
    index : bool, optional
        Tells the function whether the provided `side_length` and `center` are
        in index units or physical units. By default False, meaning they are in
        physical units.

    Returns
    -------
    np.ndarray
        The square aperture mask.
    '''
    return rect_aperture(
        x,
        y,
        width_x=side_length,
        width_y=side_length,
        center=center,
        index=index
    )

## -------------------------------------------- ##

## Phase Transformations ##
def circ_thin_lens(
    dims:tuple,
    focal_len:float,
    wavelen:float,
    index=False
):
    pass

def cyl_thin_lens(
    dims:tuple,
    focal_axis:str,
    focal_len:float,
    wavelen:float,
    index=False
):
    pass

# Do I need off-center phase transformations?

## -------------------------------------------- ##

## Wavefronts ##
def gaussian(
    x:np.ndarray,
    y:np.ndarray,
    A:float,
    waist_x:float,
    waist_y:float,
    center:tuple,
    index=False
):
    pass

def circ_gaussian(
    x:np.ndarray,
    y:np.ndarray,
    A:float,
    waist:float,
    center:tuple,
    index=False
):
    pass

def plane_wave(
    x:np.ndarray,
    y:np.ndarray,
    wavelen:float,
    theta:float=0.0,
    phi:float=0.0,
    index=False
):
    pass

## -------------------------------------------- ##

## Index of Refraction Transformations ##


## -------------------------------------------- ##