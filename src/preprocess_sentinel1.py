"""Pre-process Sentinel-1 images with the SNAP toolbox."""

from itertools import product
import os
import subprocess
import shutil
import tempfile

import numpy as np
import rasterio
import rasterio.warp
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from tqdm import tqdm

from metadata import SRC_DIR, DATA_DIR, CASE_STUDIES
from glcm import compute_textures, TEXTURES
from utils import reproject_geom
from raster import reproject, crop


def run_graph(source_product, graph, area_of_interest, output_dir):
    """Perform pre-processing in SNAP based on the GPT command-line tool.

    Parameters
    ----------
    source_product : str
        Path to source .SAFE product.
    graph : str
        Path to the graph (xml) used by GPT.
    area_of_interest : dict
        Geometry of the area of interest in a GeoJSON-like dict.
    output_dir : str
        Path to the directory where output files are written.
    """
    area_of_interest = shape(area_of_interest)
    area_of_interest = area_of_interest.wkt
    area_of_interest = area_of_interest.replace(' ((', '((')
    output_vv = os.path.join(output_dir, 'vv-gamma0.tif')
    output_vh = os.path.join(output_dir, 'vh-gamma0.tif')
    subprocess.run([
        'gpt', graph,
        '-Pgeoregion={}'.format(area_of_interest),
        '-PoutputVV={}'.format(output_vv),
        '-PoutputVH={}'.format(output_vh),
        source_product
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gamma0_computed(output_dir):
    """Check if processing to Gamma0 for both polarizations
    have been performed.
    """
    if not os.path.isdir(output_dir):
        return False
    vv = 'vv-gamma0.tif' in os.listdir(output_dir)
    vh = 'vh-gamma0.tif' in os.listdir(output_dir)
    return vv and vh


def to_gamma0(overwrite=False):
    """Pre-process all Sentinel-1 products by running the following
    SNAP operators: Apply-Orbit-File, Subset, Calibration,
    ThermalNoiseRemoval, Terrain-Flattening, Terrain-Correction.
    See the `snap.xml` file for details.
    """
    progress = tqdm(total=len(CASE_STUDIES))

    for city in CASE_STUDIES:

        source_product = os.path.join(
            DATA_DIR, 'raw', 'sentinel-1', city.sentinel_id + '.SAFE')
        output_dir = os.path.join(
            DATA_DIR, 'processed', 'sentinel-1', city.id)

        if gamma0_computed(output_dir) and not overwrite:
            progress.update(1)
            continue

        graph = os.path.join(SRC_DIR, 'snap_lee.xml')
        aoi = city.aoi['geometry']
        os.makedirs(output_dir, exist_ok=True)
        run_graph(source_product, graph, aoi, output_dir)
        progress.update(1)

    progress.close()


def textures_computed(output_dir, kind, radius, polarisation):
    """Check if the textures of a given kind (simple, advanced..) 
    and for a given radius (3x3, 5x5...) and polarisation (vv or vh)
    have been computed.
    """
    if not os.path.isdir(output_dir):
        return False
    computed = True
    images = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
    labels_computed = [f.replace('.tif', '') for f in images]
    for texture in TEXTURES[kind]:
        label = f'{polarisation.lower()}_{texture}_{radius*2+1}x{radius*2+1}'
        if label not in labels_computed:
            computed = False
    return computed


def compute_glcm_textures(polarisations, kinds, radii):
    """Compute GLCM textures for various offsets and radiuses using
    OrfeoToolbox.
    """
    progress = tqdm(total=(
        len(CASE_STUDIES) * len(polarisations) * len(kinds) * len(radii)))

    for city in CASE_STUDIES:

        output_dir = os.path.join(
            DATA_DIR, 'processed', 'sentinel-1', city.id, 'textures')
        os.makedirs(output_dir, exist_ok=True)

        # Get min. and max. values by cutting 2% the histogram of the image
        # masked by a rough urban mask (smaller AOI).
        aoi = city.aoi['features'][0]['geometry']
        aoi = shape(aoi)
        aoi = reproject_geom(aoi, src_epsg=city.epsg, dst_epsg=4326)
        aoi = mapping(aoi)

        for polarisation in polarisations:

            # Load image data both masked and unmasked
            filename = f'{polarisation}-gamma0.tif'
            img_path = os.path.join(
                DATA_DIR, 'processed', 'sentinel-1', city.id, filename)
            with rasterio.open(img_path) as src:
                profile = src.profile
                gamma0 = src.read(1)
                gamma0_masked, _ = mask(src, [aoi], crop=True, nodata=0)
                gamma0_masked = gamma0_masked[0, :, :]

            # Get new vmin and vmax values based on the rough
            # urban mask and a 2% histogram cutting
            values = gamma0_masked[gamma0_masked != 0].ravel()
            vmin = np.percentile(values, 2)
            vmax = np.percentile(values, 98)
            gamma0[gamma0 < vmin] = vmin
            gamma0[gamma0 > vmax] = vmax

            # Rescale to UINT8 range
            gamma0 = np.interp(gamma0, (vmin, vmax), (0, 255))
            gamma0 = gamma0.astype(np.uint8)
            profile.update(dtype=gamma0.dtype.name, nodata=None)

            for kind, radius in product(kinds, radii):

                if textures_computed(output_dir, kind, radius, polarisation):
                    progress.update(1)
                    continue

                compute_textures(
                    src_image=gamma0,
                    src_profile=profile,
                    dst_dir=output_dir,
                    kind=kind,
                    x_radius=radius,
                    y_radius=radius,
                    x_offset=1,
                    y_offset=1,
                    image_min=0,
                    image_max=255,
                    nb_bins=32,
                    prefix=polarisation + '_'
                )

                progress.update(1)

    progress.close()


def check_crs(img_path, crs):
    """Check if a given CRS is assigned to an image."""
    with rasterio.open(img_path) as src:
        return crs == src.crs


def to_utm(overwrite=False):
    """Reproject images to UTM after all the preprocessing steps have been
    performed. Also crop each image to avoid border noise that occured
    during textures computation by removing 100 pixels from each direction.
    """
    progress = tqdm(total=len(CASE_STUDIES))

    for city in CASE_STUDIES:

        gamma_dir = os.path.join(DATA_DIR, 'processed', 'sentinel-1', city.id)
        textures_dir = os.path.join(gamma_dir, 'textures')
        tmp_dir = tempfile.TemporaryDirectory()

        dst_crs = {'init': f'epsg:{city.epsg}'}

        for directory in [gamma_dir, textures_dir]:

            files = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith('.tif')
            ]

            for f in files:

                with rasterio.open(f) as src:
                    src_img = src.read(1)
                    src_profile = src.profile
                    src_bounds = src.bounds
                    src_crs = src.crs
                    src_affine = src.affine
                
                # If CRS is already correct, abort
                if src_crs == dst_crs and not overwrite:
                    progress.update(1)
                    continue

                dst_img, dst_affine = reproject(
                    src_img, src_crs, src_affine, src_bounds, dst_crs,
                    resampling=rasterio.warp.Resampling.cubic
                )
                dst_img, dst_affine = crop(
                    dst_img, dst_affine, n=100
                )

                dst_profile = src_profile.copy()
                dst_profile.update(
                    height=dst_img.shape[0], width=dst_img.shape[1],
                    affine=dst_affine, crs=dst_crs, transform=None
                )

                tmp_file = os.path.join(tmp_dir.name, os.path.basename(f))
                with rasterio.open(tmp_file, 'w', **dst_profile) as dst:
                    dst.write(dst_img, 1)

                # Keep original unprojected Gamma0, but remove old GLCM
                # textures to save disk space
                if 'gamma0' in f:
                    dst_f = f.replace('gamma0.tif', 'gamma0-utm.tif')
                else:
                    os.remove(f)
                    dst_f = f

                shutil.move(tmp_file, dst_f)
        
        progress.update(1)

        tmp_dir.cleanup()

    progress.close()


if __name__ == '__main__':
    print('Processing Sentinel-1 GLC data to Gamma0...')
    to_gamma0()
    print('Computing GLCM textures...')
    compute_glcm_textures(
        polarisations=['vv', 'vh'],
        kinds=['simple', 'advanced'],
        radii=[2, 3, 4, 5]
    )
    print('Reprojecting data to UTM...')
    to_utm()
