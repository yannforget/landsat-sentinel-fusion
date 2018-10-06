"""Pre-process reference shapefiles: reprojection to UTM if necessary
and rasterization.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import rasterio
import rasterio.features

from metadata import CASE_STUDIES, DATA_DIR


LAND_COVERS = [
    'builtup',
    'baresoil',
    'lowveg',
    'highveg'
]


def preprocess(city):
    """Rasterize land cover reference shapefiles for a given case study."""
    src_dir = os.path.join(DATA_DIR, 'raw', 'reference', city.id)
    dst_dir = os.path.join(DATA_DIR, 'processed', 'reference', city.id)
    os.makedirs(dst_dir, exist_ok=True)

    shapes = []
    for i, land_cover in enumerate(LAND_COVERS):
        features = gpd.read_file(os.path.join(src_dir, f'{land_cover}.shp'))
        if features.crs != city.crs:
            features.to_crs(city.crs, inplace=True)
        for geom in features.geometry:
            shapes.append((geom, i + 1))

    # Get target raster profile based on an existing SAR image
    src_img = os.path.join(
        DATA_DIR, 'processed', 'sentinel-1', city.id, 'gamma0-vh.tif')
    with rasterio.open(src_img) as src:
        dst_profile = src.profile

    # Rasterize
    dst_img = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(dst_profile['height'], dst_profile['width']),
        transform=dst_profile['affine'],
        dtype=rasterio.uint8
    )

    # Update meta profile
    dst_profile.update(
        transform=None, dtype=rasterio.uint8,
        compression='LZW', nodata=None
    )

    # Write to disk
    dst_path = os.path.join(dst_dir, 'reference.tif')
    with rasterio.open(dst_path, 'w', **dst_profile) as dst:
        dst.write(dst_img, 1)
    print(f'{city.id}/reference.tif')


if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        processing = executor.map(preprocess, CASE_STUDIES)