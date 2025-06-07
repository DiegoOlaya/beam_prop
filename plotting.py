import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from PIL import Image

## Color Functions ##

def _process_color(color) -> tuple:
    '''Process color input and return a normalized RGB tuple.
    The function accepts hex color strings, RGB tuples, and RGBA tuples. The 
    returned tuple is normalized to the range [0, 1].

    Parameters
    ----------
    color : str, listlike, tuple
        Color input to be processed. Accepts hex strings (#RRGGBB or #RGB) or 
        RGB/RGBA tuples.

    Returns
    -------
    tuple
        Normalized RGB tuple. If RGBA is provided, the alpha value is also
        included in the tuple.

    Raises
    ------
    ValueError
        If the provided hex color string is not in the correct format.
    ValueError
        If the provided string is not a hex color string.
    ValueError
        If RGB values are not in the range [0, 255].
    ValueError
        If alpha value is not in the range [0, 1].
    RuntimeError
        If the color format is not recognized.
    '''
    # Handle #hex color strings and return rgb tuple.
    if type(color) == str:
        if color.startswith('#'):
            # Execute only if type and format are correct.
            color = color.lstrip('#')
            if len(color) == 6:
                val_arr = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                return tuple([i / 255 for i in val_arr])
            # Allow the shorthand #RGB format for repeated values.
            elif len(color) == 3:
                val_arr = tuple(int(color[i]*2, 16) for i in range(3))
                return tuple([i / 255 for i in val_arr])
            else:
                raise ValueError("Invalid hex color format. Use #RRGGBB or #RGB.")
        else:
            raise ValueError("Invalid string color format. Use hex format (#RRGGBB or #RGB).")
    
    # Handle RGB
    if len(color) == 3:
        c_list = [0, 0, 0]
        for i in range(3):
            # Throw error if color values out of range.
            if color[i] < 0 or color[i] > 255:
                raise ValueError("Color values must be between 0 and 255.")
            # Normalize color values to 0-1 range if needed.
            if max(color) > 1:
                c_list[i] = color[i] / 255
            else:
                c_list[i] = color[i]
        return tuple(c_list)
    
    # Handle RGBA
    if len(color) == 4:
        c_list = [0, 0, 0, 0]
        # Process RGB components and check for errors.
        for i in range(3):
            if color[i] < 0 or color[i] > 255:
                raise ValueError("Color values must be between 0 and 255.")
            if max(color) > 1:
                c_list[i] = color[i] / 255
            else:
                c_list[i] = color[i]
        # Check if alpha value is in the correct range.
        if color[3] < 0 or color[3] > 1:
            raise ValueError("Alpha value must be between 0 and 1.")
        c_list[3] = color[3]
        return tuple(c_list)
    
    raise RuntimeError("Color format not recognized. Use hex format (#RRGGBB or #RGB) or RGB/RGBA tuple.")
    
## -------------------------------------------- ##

## Contour Plotting ##

def plot_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: list,
    max_colors: list,
    norm_method : str = 'channel',
    ax: mpl.axes.Axes = None,
    interpolation: str = 'none',
    origin: str = 'lower',
):
    '''Plot the contours of a list of 2D fields or cross-sections using the 
    matplotlib contour functions. The function uses contourf and contour to 
    plot a smooth set of contours using the given parameters.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates for the plots.
    y : np.ndarray
        Array of y-coordinates for the plots.
    z : list
        A list of 2D arrays, each representing a field array to be plotted.
    levels : int
        The number of contour levels to be plotted.
    max_colors : list
        A list of the colors to be used for each field.
    norm_method : str, optional
        The normalization method to be used for the fields, by default 
        'channel'. Must be one of 'channel', 'global', or 'equal'. 'Channel' 
        normalizes each RGB channel separately; 'global' normalizes all fields 
        by dividing them by the maximum value across all fields; and 'equal' 
        normalizes each field by dividing it by its maximum value and then
        dividing by the number of fields to ensure equal weighting.
    ax : mpl.axes.Axes, optional
        The Matplotlib Axis object to plot the fields on, by default None, 
        which creates a new figure and axis object.
    interpolation : str, optional
        Matplotlib options to be passed for imshow(), by default 'none'.
    origin : str, optional
        Where to place the [0,0] location of the image, by default 'lower'.
        This parameter accounts for possible mirroring of the image depending
        on how the physical coordinates are defined in the simulation.

    Returns
    -------
    tuple, (Figure, Axis)
        A tuple with both the Matplotlib figure and axis objects. If the Axis 
        object is passed to the function, the tuple will be of the form 
        (None, Axis).

    Raises
    ------
    ValueError
        If the number of colors does not match the number of fields.
    ValueError
        If the normalization method is not one of 'channel', 'global', or
        'equal'.
    '''
    # Error checking.
    if norm_method not in ['channel', 'global', 'equal']:
        raise ValueError("Normalization method must be 'channel', 'global', or 'equal'.")
    
    num_fields = len(z)
    if len(max_colors) != num_fields:
        raise ValueError("Number of colors must match number of fields.")
    
    # Calculate global maximum if using 'global' normalization.
    if norm_method == 'global':
        zmax = np.max([np.max(fld) for fld in z])
    
    bitmap = np.zeros((len(y), len(x), 3))
    # Plot each of the fields in succession.
    fn_iterables = zip(z, max_colors)
    fig, ax = plt.subplots() if ax is None else (None, ax)
    for z_i, col_i in fn_iterables:
        # Process the color input for the given field.
        color = _process_color(col_i)

        # Normalize field based on the specified method.
        if norm_method == 'global':
            z_i = z_i / zmax  # Normalize the field by the global maximum.
        else:
            z_i = z_i / np.max(z_i)  # Normalize the field by its maximum.
        # If the weighting is 'equal', divide by the number of fields.
        if norm_method == 'equal':
            z_i = z_i / num_fields 
        
        # Add the fields to the bitmap.
        for i in range(3):
            bitmap[:, :, i] += z_i * color[i]

    # If norm method is 'channel', normalize each RGB channel separately.
    if norm_method == 'channel':
        for i in range(3):
            if np.max(bitmap[:, :, i]) > 0:
                bitmap[:, :, i] = bitmap[:, :, i] / np.max(bitmap[:, :, i])

    # Generate the contour plot using imshow.
    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax_img = ax.imshow(
        bitmap,
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        aspect='auto',
        interpolation=interpolation,
        origin=origin,
    )
    # Return the figure and axis objects.
    return fig, ax

## -------------------------------------------- ##

## Bitmap Plotting ##

# Create bitmap from a single array.
def _make_bitmap_from_array(arr: np.ndarray) -> np.ndarray:
    '''Helper function to create a bitmap from a single array. The resulting 
    array is scaled to be between 0 and 255, with 255 always being the maximum 
    value in the original array. The resulting bitmap is an unsigned 8-bit 
    integer.

    Parameters
    ----------
    arr : np.ndarray
        A 2D numeric array to be converted to a bitmap.

    Returns
    -------
    np.ndarray
        A 2D array with the same shape as the input array, with values 
        scaled to be between 0 and 255.
    '''
    max_val = np.max(arr)
    scaled_arr = arr / max_val
    scaled_arr = (scaled_arr * 255)
    return np.rint(scaled_arr).astype(np.uint8)

# Create a bitmap from a set of up to three arrays.
def gen_rgb_bitmap(
    to_plot: list,
    channel_order: list = ['r', 'g', 'b'],
    convert: bool = True,
):
    # Error checking.
    if len(to_plot) > 3: 
        raise ValueError("A maximum of three arrays can be plotted.")
    if len(channel_order) != 3:
        raise ValueError("Must be a list of length three for each RGB channel.")
    
    # Create the bitmap.
    if convert:
        to_plot = [_make_bitmap_from_array(arr) for arr in to_plot]

    if len(to_plot) < 3:
        # Pad the list with zeros if less than 3 arrays are provided.
        to_plot += [np.zeros_like(to_plot[0])] * (3 - len(to_plot))

    # Reorder the arrays according to the given channel order.
    to_plot = [to_plot[channel_order.index(c)] for c in ['r', 'g', 'b']]

    # Flatten and then produce the RGB channel array.
    to_plot_flat = [arr.flatten() for arr in to_plot]
    rgb_flt = [(r, g, b) for r, g, b in zip(*to_plot_flat)]
    rgb_arr = np.array(rgb_flt, dtype=np.uint8)
    # Reshape the array to the original shape.
    rgb_shape = to_plot[0].shape
    rgb_arr = rgb_arr.reshape((*rgb_shape, 3))
    # Return final RGB array.
    return rgb_arr

# Generate an image from a bitmap.
def make_bit_img(
    bitmap: np.ndarray,
    origin: str = 'lower',
):
    '''Generates a matplotlib image from a bitmap array. This is a wrapper for 
    plt.imshow().

    Parameters
    ----------
    bitmap : np.ndarray
        An array of shape (n, m, 3) representing and RGB image.
    origin : str, optional
        Where to place the [0,0] location of the image, by default 'lower'. 
        This can be 'upper' or 'lower', depending on how the image should be 
        displayed.

    Returns
    -------
    plt.AxesImage
        A matplotlib image object displaying the bitmap image.
    '''
    return plt.imshow(bitmap, origin=origin)

# Save a bitmap as an image file.
def save_bitmap_img(
    bitmap: np.ndarray,
    filename: str,
    fmt: str = None,
):
    '''Saves a bitmap array as an image file using the provided filename and 
    format. Note that the the bitmap saves in image coordinates, which places 
    the [0,0] location in the upper left corner of the image. Depending on how
    the physical coordinates are defined in the simulation, this may result in 
    mirrored images.

    Parameters
    ----------
    bitmap : np.ndarray
        RGB array of shape (n, m, 3) representing the image to be saved.
    filename : str
        The name of the file to save the image to.
    fmt : str, optional
        The format to save the image in, by default None. If None, the format 
        is inherited from the filename. Format options are those supported by
        pillow.

    Raises
    ------
    ValueError
        If the filename does not include a file extension when no format is 
        specified.
    ValueError
        If the specified format does not match the filename extension.
    '''
    img = Image.fromarray(bitmap, mode='RGB')
    if fmt is not None:
        if '.' in filename:
            ext_fmt = filename.split('.')[-1].upper()
            if ext_fmt != fmt:
                raise ValueError("Filename format does not match specified format.")
        # Save the image with the given format.
        img.save(filename, format=fmt)
        return
    
    # Format not specified.
    if '.' not in filename:
        raise ValueError("Filename must include a file extension if no format is specified.")
    img.save(filename)
    return

## -------------------------------------------- ##