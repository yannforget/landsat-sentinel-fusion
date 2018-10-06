"""Raster data processing."""

import numpy as np
import rasterio
import rasterio.warp


def histogram_cutting(raster, percent=2, nodata=None, mask=None):
    """Perform histogram cutting on a 2D raster

    Parameters
    ----------
    raster : numpy 2d array
        Input raster.
    percent : int
        Percentile (default=2).
    nodata : int or float, optional
        Nodata value of the input raster.
    mask : numpy 2d array, optional
        Masked pixel values will be ignored.

    Returns
    -------
    output : numpy 2d array
        Output raster.
    """
    output = raster.copy()
    if nodata:
        output[output == nodata] = np.nan
    if isinstance(mask, np.ndarray):
        output[mask] = np.nan
    vmin, vmax = np.percentile(
        output[~np.isnan(output)].ravel(), (percent, 100-percent))
    output[(output < vmin)] = vmin
    output[(output > vmax)] = vmax
    return output


def rescale(values, dst_min, dst_max):
    """Rescale the values of an array to a new range.

    Parameters
    ----------
    values : numpy 1d array
        Input values.
    dst_min : int or float
        New min. value.
    dst_max : int or float
        New max. value.

    Returns
    -------
    values : numpy 1d array
        Output values.
    """
    num = (dst_max - dst_min) * (values - values.min())
    den = values.max() - values.min()
    return (num / den) + dst_min


def rescale_raster(raster, dst_min, dst_max, nodata=None):
    """Rescale the values of a 2D raster to a new range.

    Parameters
    ----------
    raster : numpy 2d array
        Input raster.
    dst_min : int or float
        Target min. value.
    dst_max : int or float
        Target max. value.
    nodata : int or float, optional
        Nodata value in the input raster.

    Returns
    -------
    output : numpy 2d array
        Output raster.
    """
    if nodata:
        mask = (raster != nodata) & ~np.isnan(raster)
    else:
        mask = ~np.isnan(raster)
    output = raster.copy()
    values = raster[mask]
    values = rescale(values, dst_min, dst_max)
    output[mask] = values
    return output


def reproject(src_img, src_crs, src_affine, src_bounds, dst_crs, resampling):
    """Reproject an image to a given CRS.

    Parameters
    ----------
    src_img : numpy 2d array
        Source image as a 2d numpy array.
    src_crs : dict
        Source CRS.
    src_affine : Affine
        Source affine.
    src_bounds : tuple
        Source bounds (left, bottom, right, top).
    dst_crs : dict
        Target EPSG.
    resampling : Resampling
        Resampling method provided with a rasterio.warp.Resampling object.

    Returns
    -------
    dst_img : numpy 2d array
        Output reprojected image.
    dst_affine : Affine
        Output updated affine.
    """
    # If dst_crs or src_crs are provided as integer EPSG, convert to dict
    if isinstance(src_crs, int):
        src_crs = {'init': f'epsg:{src_crs}'}
    if isinstance(dst_crs, int):
        dst_crs = {'init': f'epsg:{dst_crs}'}

    # Caculate the new affine and shape
    src_height, src_width = src_img.shape
    dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds
    )

    # Get the reprojected image
    dst_img = np.ndarray(shape=(dst_height, dst_width), dtype=src_img.dtype)
    rasterio.warp.reproject(
        source=src_img, destination=dst_img,
        src_transform=src_affine, dst_transform=dst_affine,
        src_crs=src_crs, dst_crs=dst_crs,
        resampling=resampling
    )

    return dst_img, dst_affine


def crop(src_img, src_affine, n):
    """Crop a given image from each direction according to a given number of
    pixels. Also calculate a new Affine transformation.

    Parameters
    ----------
    src_img : numpy 2d array
        Source image as a 2d numpy array.
    src_affine : Affine
        Source Affine object.
    n : int
        Number of pixels cropped from each direction.

    Returns
    -------
    dst_img : numpy 2d array
        Output cropped image.
    dst_affine : Affine
        Updated affine.
    """
    nrows, ncols = src_img.shape
    dst_img = src_img[n:nrows-n, n:ncols-n]
    a, b, c, d, e, f, _, _, _ = src_affine
    c += a * n
    f -= a * n
    dst_affine = rasterio.Affine(a, b, c, d, e, f)
    return dst_img, dst_affine
