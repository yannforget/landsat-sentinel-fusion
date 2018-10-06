"""Pre-processing Landsat 8 imagery: resample surface reflectance products
for homogeneity with Sentinel-1 data.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
from shapely.geometry import box, mapping
from tqdm import tqdm

from metadata import CASE_STUDIES, DATA_DIR
from utils import reproject_geom, get_epsg


# Band suffixes (from LaSRC) and their labels
BANDS = [
    ('sr_band2', 'blue'),
    ('sr_band3', 'green'),
    ('sr_band4', 'red'),
    ('sr_band5', 'nir'),
    ('sr_band6', 'swir1'),
    ('sr_band7', 'swir2'),
    ('bt_band10', 'tir1'),
    ('bt_band11', 'tir2'),
    ('cfmask', 'cfmask')
]


def resample(src_raster, dst_affine, dst_crs, dst_shape, dst_bounds):
    """Resample, crop and reproject an input raster according
    to a target profile and bounds.

    Parameters
    ----------
    src_raster : str
        Path of the input raster.
    dst_affine : Affine
        Target affine.
    dst_crs : dict
        Target CRS.
    dst_shape : tuple
        Target shape (height, width).
    dst_bounds : tuple
        Target boundaries in CRS of `dst_profile`.
        Format is (left, bottom, right, top).

    Returns
    -------
    dst_img : numpy 2d array
        Output raster.
    """
    dst_epsg = get_epsg(dst_crs)

    with rasterio.open(src_raster) as src:

        # Create shapely geometry from bounds and reproject it
        # to the CRS of `src_raster`. Then convert it to a
        # GeoJSON-like dictionnary.
        src_epsg = get_epsg(src.crs)
        geom = box(*dst_bounds)
        geom = reproject_geom(geom, dst_epsg, src_epsg)
        geom = mapping(geom)

        # Load masked `src_data`
        src_profile = src.profile
        src_img, src_affine = rasterio.mask.mask(
            src, [geom], crop=True)
        src_img = src_img[0, :, :]

    # Resample & Reproject
    dst_img = np.ndarray(shape=dst_shape, dtype=src_img.dtype)
    rasterio.warp.reproject(
        source=src_img, destination=dst_img,
        src_transform=src_affine, dst_transform=dst_affine,
        src_crs=src_profile['crs'], dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.cubic
    )

    return dst_img


def preprocess(city):
    """Mask, resample & reproject Landsat 8 data of a given case study
    based on an existing Sentinel-1 image.

    Parameters
    ----------
    city : Metadata
        Metadata object of a given case study.
    """
    src_dir = os.path.join(
        DATA_DIR, 'raw', 'landsat', city.landsat_id)
    dst_dir = os.path.join(
        DATA_DIR, 'processed', 'landsat', city.id)
    sar_img = os.path.join(
        DATA_DIR, 'processed', 'sentinel-1',
        city.id, 'textures', 'vh_entropy_5x5.tif')
    os.makedirs(dst_dir, exist_ok=True)

    # Get master profile from SAR image
    with rasterio.open(sar_img) as src:
        dst_profile = src.profile
        dst_bounds = src.bounds
    dst_affine = dst_profile['affine']
    dst_crs = dst_profile['crs']
    dst_shape = (dst_profile['height'], dst_profile['width'])

    for suffix, name in BANDS:

        src_path = os.path.join(src_dir, f'{city.landsat_id}_{suffix}.tif')
        dst_path = os.path.join(dst_dir, f'{name}.tif')

        print(f'{city.id}/{name}.tif')

        # Abort if dst_path already exists
        if os.path.isfile(dst_path):
            continue

        img = resample(src_path, dst_affine, dst_crs, dst_shape, dst_bounds)

        if img.dtype.name == 'int16':
            nodata = -9999
        else:
            nodata = None

        # Update meta profile
        dst_profile.update(
            transform=None, dtype=img.dtype.name,
            compression='LZW', nodata=nodata
        )

        # Write to disk
        with rasterio.open(dst_path, 'w', **dst_profile) as dst:
            dst.write(img, 1)


if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        processing = executor.map(preprocess, CASE_STUDIES)
