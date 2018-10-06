"""Generate 40x40km areas of interest for each case study."""

import os
import json

from shapely.geometry import Point, mapping

from metadata import Metadata, CASE_STUDIES, DATA_DIR
import utils


def buffer_extent(lat, lon, dst_epsg, buffer_size):
    """Generate a buffer around a location and returns
    the geometry corresponding to its spatial envelope.
    """
    center = utils.reproject_geom(
        Point(lon, lat), src_epsg=4326, dst_epsg=dst_epsg)
    buffer = center.buffer(buffer_size)
    return buffer.exterior.envelope


def area_of_interest(lat, lon, buffer_size):
    """Get the shapely geometry corresponding to the area
    of interest.
    """
    intermediary_epsg = utils.find_utm_epsg(lat, lon)
    extent = buffer_extent(lat, lon, intermediary_epsg, buffer_size)
    aoi = utils.reproject_geom(extent, intermediary_epsg, 4326)
    return aoi


def write_as_geojson(geom, out_f):
    """Write the geometry as GeoJSON."""
    geojson = {'type': 'Feature', 'geometry': mapping(geom)}
    with open(out_f, 'w') as f:
        json.dump(geojson, f, indent=True)
    return


if __name__ == '__main__':
    for city in CASE_STUDIES:
        out_dir = os.path.join(DATA_DIR, 'processed', 'masks', city.id)
        os.makedirs(out_dir, exist_ok=True)
        out_f = os.path.join(out_dir, 'aoi.geojson')
        aoi = area_of_interest(city.lat, city.lon, buffer_size=20000)
        write_as_geojson(aoi, out_f)
