"""Utility functions."""

import math
from functools import partial

import pyproj
from shapely.geometry import shape
from shapely.ops import transform


def find_utm_epsg(lat, lon):
    """Find the UTM EPSG based on lat/lon coordinates."""
    utm_zone = (math.floor((lon + 180) // 6) % 60) + 1
    if lat >= 0:
        pole = 600
    else:
        pole = 700
    return 32000 + pole + utm_zone


def get_epsg(crs):
    """Get EPSG code from a CRS dictionnary."""
    epsg = crs['init'].split(':')[-1]
    return int(epsg)


def get_crs(epsg):
    """Get CRS dictionnary from an EPSG code."""
    return {'init': f'epsg:{epsg}'} 


def reproject_geom(geom, src_epsg, dst_epsg):
    """Reproject a shapely geometry given a source EPSG and a
    target EPSG.
    """
    src_proj = pyproj.Proj(init='epsg:{}'.format(src_epsg))
    dst_proj = pyproj.Proj(init='epsg:{}'.format(dst_epsg))
    reproj = partial(pyproj.transform, src_proj, dst_proj)
    return transform(reproj, geom)


def named_crs(epsg):
    """Construct a named CRS dict given an EPSG code
    according to the GeoJSON spec.
    """
    return {
        'type': 'name',
        'properties': {
            'name': 'urn:ogc:def:crs:EPSG::{}'.format(epsg)
        }
    }


def get_geometry(geojson, dst_epsg=4326):
    """Get the geometry contained in a GeoJSON file
    as a shapely geometry. If a `FeatureCollection`,
    only the first feature is returned.
    
    Parameters
    ----------
    geojson : dict
        GeoJSON-like dict (Feature or Feature Collection).
    dst_epsg : int
        Target EPSG code.
    
    Returns
    -------
    geometry : shapely geometry
        First GeoJSON feature as a shapely geometry.
    """
    if geojson['type'] == 'FeatureCollection':
        geom_dict = geojson['features'][0]['geometry']
    else:
        geom_dict = geojson['geometry']
    geometry = shape(geom_dict)

    src_epsg = 4326
    if 'crs' in geojson:
        crs = geojson['crs']['properties']['name']
        src_epsg = crs.split(':')[-1]
        src_epsg = int(src_epsg)
    
    if src_epsg != dst_epsg:
        geometry = reproject_geom(geometry, src_epsg, dst_epsg)
    
    return geometry
