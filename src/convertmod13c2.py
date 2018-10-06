"""Convert MOD13C2 HDF subdatasets to GeoTIFF and reproject them
to EPSG:4326.
"""

import os
import subprocess

from metadata import DATA_DIR


def main():
    """Save each .HDF subdataset as an individual GeoTIFF file
    reprojected to EPSG:4326.
    """
    MOD13C2_DIR = os.path.join(DATA_DIR, 'raw', 'MOD13C2')
    SUBDATASET = 'MOD_Grid_monthly_CMG_VI:CMG 0.05 Deg Monthly NDVI'
    DRIVER = 'HDF4_EOS:EOS_GRID'

    hdf_files = [f for f in os.listdir(MOD13C2_DIR) if f.endswith('.hdf')]
    os.chdir(MOD13C2_DIR)

    for i, hdf in enumerate(sorted(hdf_files)):
        month = str(i+1).zfill(2)
        subprocess.run([
            'gdal_translate', '-of', 'GTiff',
            '{}:"{}":{}'.format(DRIVER, hdf, SUBDATASET),
            'mod13c2_{}.tif'.format(month)
        ])


if __name__ == '__main__':
    main()
