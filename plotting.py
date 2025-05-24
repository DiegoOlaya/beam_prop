import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

## ---- Colormap Functions ---- ##

def gen_rgb_cmap(
    color = 'red', 
    name = 'rgb_scale', 
    n_levels = 256
) -> mpl.colors.ListedColormap:
    '''Return a gradient colormap for one of the three possible RGB colors.
    The colormap is a gradient from black to the specified color.

    Parameters
    ----------
    color : str, optional
        The RGB color for the returned colormap, either 'red', 'green', 
        or 'blue', by default 'red'
    name : str, optional
        A string with a name descrbing the resulting colormap, 
        by default 'rgb_scale'
    n_levels : int, optional
        The integer number of levels for the resulting colormap, by 
        default 256.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Returns a leveled colormap of the specified color, where the number of 
        levels is given by n_levels. The colormap is starts at black and ends
        at 255 for the given color.

    Raises
    ------
    ValueError
        If the color is not 'red', 'green', or 'blue'.
    ValueError
        If the number of levels is not a positive integer.
    '''
    if color not in ['red', 'green', 'blue']:
        raise ValueError("Color must be 'red', 'green', or 'blue'.")
    if n_levels < 1:
        raise ValueError("Number of levels must be a positive integer.")
    
    lvls = np.linspace(0, 255, n_levels)
    if color == 'red':
        colors = np.array([lvls / 255, np.zeros_like(lvls), np.zeros_like(lvls)]).T
    elif color == 'green':
        colors = np.array([np.zeros_like(lvls), lvls / 255, np.zeros_like(lvls)]).T
    elif color == 'blue':
        colors = np.array([np.zeros_like(lvls), np.zeros_like(lvls), lvls / 255]).T
    
    cmap = mpl.colors.ListedColormap(colors, name=name)
    return cmap

def gen_camp_from_color(
    end_color,
    start_color = (0, 0, 0, 1),
    name = 'custom_cmap', 
    n_levels = 256
) -> mpl.colors.LinearSegmentedColormap:
    '''Returns a gradient colormap between the two specified colors.

    Parameters
    ----------
    end_color : str, listlike, tuple
        The color corresponding to the maximum value of the colormap.
    start_color : str, listlike, or tuple; optional
        The color corresponding to the minimum value of the colormap, by 
        default (0, 0, 0), representing black.
    name : str, optional
        A name for the colormap to provide to the matplotlib method, by default
        'custom_cmap'
    n_levels : int, optional
        The number of steps in the the level colormap, by default 256

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A colormap object with a gradient between the two specified colors.
    '''
    e_col = _process_color(end_color)
    s_col = _process_color(start_color)
    c_map = mpl.colors.LinearSegmentedColormap.from_list(
        name,
        [s_col, e_col],
        N=n_levels,
    )
    return c_map

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
    
## ----------------------------- ##

## ---- Matplotlib Contour Plotting ---- ##

def plot_contours(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    levels: int,
    cmap: mpl.colors.ListedColormap,
    alpha_f: float = 1.0,
    alpha_c: float = 1.0,
    ax: mpl.axes.Axes = None,
    antialiased: bool = True,
    linewidths: float = 0.5,
):
    '''Plot the contours of a 2D field using matplotlib functions. Uses both
    contourf and contour to plot the filled and line contours, respectively, 
    avoiding white spaces in the plots when alpha is less than 1.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates for the plots.
    y : np.ndarray
        Array of y-coordinates for the plots.
    z : np.ndarray
        Array of values for each xy coordinate pair.
    levels : int
        The number of contour levels to be plotted.
    cmap : mpl.colors.ListedColormap
        The color map to be used in the plots.
    alpha_f : float, optional
        The transparency values for the filled contour plot, by default 1.0
    alpha_c : float, optional
        The transparency values for the edge contour plot, by default 1.0
    ax : mpl.axes.Axes, optional
        The axis object on which to plot the contours, by default None, which 
        creates a new figure and axis object.
    antialiased : bool, optional
        Matplotlib keyword argument passed to the filled contour plot, by 
        default True.
    linewidths : float, optional
        The width of the drawn contour lines, by default 0.5

    Returns
    -------
    tuple, (Figure, Axis)
        A tuple with both the Matplotlib figure and axis objects. If the Axis 
        object is passed to the function, the tuple will be of the form 
        (None, Axis).
    '''
    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax.contourf(
        x, y, z, levels=levels, 
        cmap=cmap, alpha=alpha_f,
        antialiased=antialiased,
    )
    ax.contour(
        x, y, z, levels=levels,
        cmap=cmap, alpha=alpha_c,
        linewidths=linewidths,
    )
    return fig, ax

def cplot_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: list,
    levels: int,
    cmaps: list,
    alphas: list,
    ax: mpl.axes.Axes = None,
    antialiased: bool = True,
    linewidths: float = 0.5,
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
    cmaps : list
        A list of colormaps to be used for each field array to be plotted.
    alphas : list
        A list of arrays with the alpha values for the filled and edge contours
        for each field. Each entry should be a list of two values, with the 
        first values representing the filled contour alpha and the second value
        representing the edge contour alpha.
    ax : mpl.axes.Axes, optional
        The Matplotlib Axis object to plot the fields on, by default None, 
        which creates a new figure and axis object.
    antialiased : bool, optional
        Matplotlib keyword argument passed to the filled contour plot, by 
        default True.
    linewidths : float, optional
        The width of the drawn contour lines, by default 0.5

    Returns
    -------
    tuple, (Figure, Axis)
        A tuple with both the Matplotlib figure and axis objects. If the Axis 
        object is passed to the function, the tuple will be of the form 
        (None, Axis).

    Raises
    ------
    ValueError
        If the number of colormaps does not match the number of fields.
    ValueError
        If any of the alpha value arrays are not of length 2.
    ValueError
        If any alpha value is not between 0 and 1.
    '''
    num_fields = len(z)
    if len(cmaps) != num_fields:
        raise ValueError("Number of colormaps must match number of fields.")
    # Check the length of the alpha list.
    if type(alphas[0]) != list:
        # Make the list the right length if all the alphas are the same.
        alphas = [alphas] * num_fields
    # Check the alpha values for all entries.
    for a_lst in alphas:
        if len(a_lst) != 2:
            raise ValueError("Alpha values must be a list of two values.")
        if a_lst[0] < 0 or a_lst[0] > 1:
            raise ValueError("Alpha values must be between 0 and 1.")
        if a_lst[1] < 0 or a_lst[1] > 1:
            raise ValueError("Alpha values must be between 0 and 1.")
    
    
    # Plot each of the fields in succession.
    fn_iterables = zip(z, cmaps, alphas)
    fig, ax = plt.subplots() if ax is None else (None, ax)
    for z_i, cmap_i, alpha_i in fn_iterables:
        af = alpha_i[0]
        ac = alpha_i[1]
        ax.contourf(
            x, y, z_i, levels=levels,
            cmap=cmap_i, alpha=af,
            antialiased=antialiased,
        )
        ax.contour(
            x, y, z_i, levels=levels,
            cmap=cmap_i, alpha=ac,
            linewidths=linewidths,
        )
    return fig, ax

## ----------------------------- ##