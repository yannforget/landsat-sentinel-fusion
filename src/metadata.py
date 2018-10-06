"""Access metadata."""


import json
import os

import pandas as pd
import rasterio

from utils import find_utm_epsg

CITIES = [
    'antananarivo',
    'bukavu',
    'chimoio',
    'dakar',
    'gao',
    'johannesburg',
    'kampala',
    'katsina',
    'nairobi',
    'ouagadougou',
    'saint_louis',
    'windhoek'
]

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')


class Metadata(object):
    def __init__(self, case_study):
        """Access city-level metadata."""
        self.id = case_study
        self.name = case_study.replace('_', '-').title()
        self.meta = self.read_csv()
        self.bounds = self.get_bounds()
        self.profile = self.get_profile()
        self.width = self.profile['width']
        self.height = self.profile['height']
        self.affine = self.profile['affine']

    def read_csv(self):
        """Read metadata CSV."""
        data = pd.read_csv(
            os.path.join(DATA_DIR, 'case-studies.csv'),
            index_col=0
        )
        return data.loc[self.id]

    @property
    def lat(self):
        return self.meta.latitude

    @property
    def lon(self):
        return self.meta.longitude

    @property
    def landsat_id(self):
        return self.meta.landsat_id

    @property
    def sentinel_id(self):
        return self.meta.sentinel_id

    @property
    def epsg(self):
        return find_utm_epsg(self.lat, self.lon)
    
    @property
    def crs(self):
        return {'init': f'epsg:{self.epsg}'}
    
    @property
    def aoi(self):
        path = os.path.join(
            DATA_DIR, 'processed', 'masks', self.id, 'aoi.geojson'
        )
        with open(path) as f:
            return json.load(f)
    
    @property
    def water(self):
        path = os.path.join(
            DATA_DIR, 'processed', 'masks', self.id, 'water.tif'
        )
        with rasterio.open(path) as src:
            return src.read(1) == 1
    
    def get_profile(self):
        sample_img = os.path.join(DATA_DIR, 'processed', 'landsat', self.id, 'blue.tif')
        with rasterio.open(sample_img) as src:
            return src.profile
    
    def get_bounds(self):
        sample_img = os.path.join(DATA_DIR, 'processed', 'landsat', self.id, 'blue.tif')
        with rasterio.open(sample_img) as src:
            return src.bounds


CASE_STUDIES = [Metadata(city) for city in CITIES]
