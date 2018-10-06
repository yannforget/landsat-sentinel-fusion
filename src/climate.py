"""Get monthly NDVI and precipitations for each case study."""

import os
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm

from metadata import CASE_STUDIES, DATA_DIR


def monthly_precipitations():
    """Get monthly precipitations based on WorldClim data."""
    worldclim_dir = os.path.join(DATA_DIR, 'raw', 'worldclim')
    months = [m for m in range(1, 13)]
    cities = [city.id for city in CASE_STUDIES]
    precipitations = pd.DataFrame(index=cities, columns=months)
    for city, month in product(CASE_STUDIES, months):
        month_str = str(month).zfill(2)
        raster_path = os.path.join(
            worldclim_dir, 'wc2.0_10m_prec_{}.tif'.format(month_str))
        with rasterio.open(raster_path) as src:
            # Read pixel corresponding to the city lat/lon coordinates
            mm = [x for x in src.sample([(city.lon, city.lat)])][0][0]
        precipitations.at[(city.id, month)] = mm
    return precipitations


def monthly_ndvi():
    """Get monthly NDVI from MOD13C2 products."""
    mod13c2_dir = os.path.join(DATA_DIR, 'raw', 'MOD13C2')
    months = [m for m in range(1, 13)]
    cities = [city.id for city in CASE_STUDIES]
    ndvi = pd.DataFrame(index=cities, columns=months)
    for city, month in product(CASE_STUDIES, months):
        aoi = city.aoi['geometry']
        raster_path = os.path.join(
            mod13c2_dir, 'mod13c2_{}.tif'.format(str(month).zfill(2))
        )
        # Mask MOD13C2 data based on the AOI
        with rasterio.open(raster_path) as src:
            data, _ = mask(src, [aoi], crop=True)
        pixels = data[0, :, :].ravel()
        # MOD13C2 valid range: -2000 - 10000
        pixels = pixels[pixels >= -2000]
        pixels = pixels[pixels <= 10000]
        # MOD13C2 scale factor: 0.0001
        pixels = pixels.astype(np.float)
        pixels = pixels * 0.0001
        ndvi.at[(city.id, month)] = pixels.mean()
    return ndvi


def daily_precipitations():
    """Get daily precipitations for each case study based on the
    CPC Global Precipitations dataset.
    """
    cpc_dir = os.path.join(DATA_DIR, 'raw', 'cpc_global_precip')
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2017, 12, 31)
    precipitations = pd.DataFrame(
        index=[city.id for city in CASE_STUDIES],
        columns=pd.date_range(start_date, end_date, freq='D'))
    progress = tqdm(total=len(CASE_STUDIES)*3)
    os.chdir(cpc_dir)
    for city in CASE_STUDIES:
        with rasterio.open('NETCDF:precip.2015.nc:"precip"') as src:
            x, y = src.index(city.lon, city.lat)
            precip2015 = src.read()[:, x, y]
            progress.update(1)
        with rasterio.open('NETCDF:precip.2016.nc:"precip"') as src:
            x, y = src.index(city.lon, city.lat)
            precip2016 = src.read()[:, x, y]
            progress.update(1)
        with rasterio.open('NETCDF:precip.2017.nc:"precip"') as src:
            x, y = src.index(city.lon, city.lat)
            precip2017 = src.read()[:, x, y]
            progress.update(1)
        precip = np.concatenate([precip2015, precip2016, precip2017])
        precip[precip < 0] = 0
        precipitations.loc[city.id] = precip
    return precipitations


if __name__ == '__main__':
    mprecipitations = monthly_precipitations()
    mprecipitations.to_csv(os.path.join(
        DATA_DIR, 'monthly-precipitations.csv'))
    ndvi = monthly_ndvi()
    ndvi.to_csv(os.path.join(DATA_DIR, 'monthly-ndvi.csv'))
    dprecipitations = daily_precipitations()
    dprecipitations.to_csv(os.path.join(
        DATA_DIR, 'daily-precipitations.csv'))
