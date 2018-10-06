"""Use water polygons from openstreetmapdata.org to mask
seas and oceans, and `natural=water|wetland` polygons
from the OSM database.
"""

import os
import shutil

import requests
import fiona
import geopandas as gpd
import numpy as np
import psycopg2
import rasterio.features
from shapely.geometry import shape
from tqdm import tqdm
from appdirs import user_data_dir

from metadata import CASE_STUDIES, DATA_DIR


URL = 'http://data.openstreetmapdata.com/water-polygons-split-4326.zip'


def download():
    """Download water-polygons shapefile."""
    dst_dir = user_data_dir(appname='osmxtract')
    os.makedirs(dst_dir, exist_ok=True)
    filename = URL.split('/')[-1]
    dst_file = os.path.join(dst_dir, filename)
    r = requests.head(URL)
    content_length = int(r.headers['Content-Length'])
    progress = tqdm(total=content_length, unit='B', unit_scale=True)
    chunk_size = 1024 ** 2
    with requests.get(URL, stream=True) as r:
        with open(dst_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(chunk_size)


def is_downloaded():
    """Check if seas shapefile is downloaded."""
    data_dir = user_data_dir(appname='osmxtract')
    expected_path = os.path.join(
        data_dir, 'water-polygons-split-4326.zip'
    )
    return os.path.isfile(expected_path)


def clean():
    """Clean downloaded data."""
    data_dir = user_data_dir(appname='osmxtract')
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)


def get_sea_polygons(bounds):
    """Get sea polygons according to the provided bounds.

    Parameters
    ----------
    bounds : tuple
        Bounds decimal lat/lon coordinates (xmin, ymin, xmax, ymax).

    Returns
    -------
    feature : iterable
        Output features as an iterable of GeoJSON-like dicts.
    """
    data_dir = user_data_dir(appname='osmxtract')
    if not is_downloaded():
        download()
    zip_path = os.path.join(data_dir, 'water-polygons-split-4326.zip')
    shp_path = '/water-polygons-split-4326/water_polygons.shp'
    with fiona.open(shp_path, vfs=f'zip://{zip_path}') as src:
        features = [feature for _, feature in src.items(bbox=bounds)]
    return features


def get_water_bodies(db, case_study_name, values=['water', 'wetland']):
    """Get the geometries with `water` or `wetland` values for the key
    `natural` in the OSM database for a given case study.
    """
    query = f"""
    SELECT
      osm_polygon.way AS geom, osm_polygon.natural AS value
    FROM
      osm_polygon, datafusion
    WHERE
      osm_polygon.natural IN ('water', 'wetland')
    AND
      datafusion.name = '{case_study_name}'
    AND
      ST_Intersects(osm_polygon.way, datafusion.geom)
    """
    water_bodies = gpd.read_postgis(query, db)
    water_bodies.crs = {'init': 'epsg:4326'}
    return water_bodies


def get_sea_raster(bounds, height, width, crs, affine):
    """Get binary water mask (sea and ocean only) according to
    the provided bounds.

    Parameters
    ----------
    bounds : tuple
        Bounds decimal lat/lon coordinates (xmin, ymin, xmax, ymax).
    height : int
        Raster height.
    width : int
        Raster width.
    crs : dict
        Target CRS.
    affine : Affine
        Target affine transformation.

    Returns
    -------
    water : numpy 2d array
        Water binary mask as a 2D numpy array.
    """
    features = get_sea_polygons(bounds)
    if len(features) > 0:
        geodataframe = gpd.GeoDataFrame.from_features(features)
        geodataframe.crs = {'init': 'epsg:4326'}
        geodataframe = geodataframe.to_crs(crs)
        geoms = ((geom, 1) for geom in geodataframe.geometry)
        return rasterio.features.rasterize(
            shapes=geoms, fill=0, all_touched=True,
            transform=affine, out_shape=(height, width),
            dtype=np.uint8
        )
    else:
        return np.zeros(shape=(height, width), dtype=np.uint8)


def get_water_bodies_raster(db, case_study_name, height, width, crs, affine):
    """Get binary water mask (water bodies + wetland) for
    a given case study.

    Parameters
    ----------
    db : db connection
        OSM database connection.
    case_study_name : str
        Case study ID.
    height : int
        Target raster height.
    width : int
        Target raster width.
    crs : dict
        Target raster CRS.
    affine : Affine
        Target raster affine transformation.
    
    Returns
    -------
    water : numpy 2d array
        Water binary mask as a 2d numpy array.
    """
    features = get_water_bodies(db, case_study_name)
    features = features.to_crs(crs)
    if len(features) > 0:
        return rasterio.features.rasterize(
            shapes=((geom, 1) for geom in features.geometry),
            transform=affine,
            out_shape=(height, width),
            dtype=np.uint8)
    else:
        return np.zeros(shape=(height, width), dtype=np.uint8)


if __name__ == '__main__':
    db = psycopg2.connect(
        database='osm',
        user='maupp',
        password='maupp',
        host='localhost'
    )
    for case_study in CASE_STUDIES:
        print(f'Processing {case_study.name}...')
        aoi = shape(case_study.aoi['geometry'])
        sea = get_sea_raster(
            bounds=aoi.bounds,
            height=case_study.height,
            width=case_study.width,
            crs=case_study.crs,
            affine=case_study.affine
        )
        water_bodies = get_water_bodies_raster(
            db=db, case_study_name=case_study.id,
            height=case_study.height,
            width=case_study.width,
            affine=case_study.affine,
            crs=case_study.crs
        )
        mask = np.max([sea, water_bodies], axis=0)
        dst_dir = os.path.join(DATA_DIR, 'processed', 'masks', case_study.id)
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, 'water.tif')
        dst_profile = case_study.profile.copy()
        dst_profile.update(dtype=np.uint8, compression='LZW', nodata=None)
        with rasterio.open(dst_file, 'w', **dst_profile) as dst:
            dst.write(mask, 1)
    db.close()