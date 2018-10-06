"""Compute simple, advanced and higher-order Haralick textures with Orfeo Toolbox.
https://www.orfeo-toolbox.org/CookBook/Applications/app_HaralickTextureExtraction.html
"""

import os
import subprocess
import tempfile

import numpy as np
import rasterio

TEXTURES = {
    'simple': [
        'energy',
        'entropy',
        'correlation',
        'inverse_difference_moment',
        'inertia',
        'cluster_shade',
        'cluster_prominence',
        'haralick_correlation'
    ],
    'advanced': [
        'mean',
        'variance',
        'dissimilarity',
        'sum_average',
        'sum_variance',
        'sum_entropy',
        'difference_of_entropies',
        'difference_of_variances',
        'ic1',
        'ic2'
    ],
    'higher': [
        'short_run_emphasis',
        'long_run_emphasis',
        'grey_level_nonuniformity',
        'run_length_nonuniformity',
        'run_percentage',
        'low_grey_level_run_emphasis',
        'high_grey_level_run_emphasis',
        'short_run_low_grey_level_emphasis',
        'short_run_high_grey_level_emphasis',
        'long_run_low_grey_level_emphasis',
        'long_run_high_grey_level_emphasis'
    ]
}


def compute_textures(
        src_image,
        src_profile,
        dst_dir,
        kind='simple',
        x_radius=5,
        y_radius=5,
        x_offset=1,
        y_offset=1,
        image_min=0,
        image_max=255,
        nb_bins=64,
        prefix='',
):
    """Compute Haralick textures. Images are written in individual
    files in `dst_dir`.

    Parameters
    ----------
        src_image : numpy 2d array
            Source image. Data type must be UINT8.
        src_profile : dict
            Rasterio profile of the source image.
        dst_dir : str
            Directory where the textures will be saved.
        kind : str, optional
            'simple', 'advanced', or 'higher' (default='simple').
        x_radius : int, optional
            X Radius (default=5).
        y_radius : int, optional
            Y Radius (default=5).
        x_offset : int, optional
            X Offset (default=1).
        y_offset : int, optional
            Y Offset (default=1).
        image_min : int, optional
            Image Minimum (default=0).
        image_max : int, optional
            Image Maximum (default=255).
        nb_bins : int, optional
            Histogram numbers of bins (default=64).
        prefix : str, optional
            Prefix to texture filenames (default='').
    """
    # Write input image to disk
    profile = src_profile.copy()
    profile.update(dtype='uint8', transform=None, count=1)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_image = os.path.join(tmp_dir.name, 'image.tif')
    with rasterio.open(tmp_image, 'w', **profile) as dst:
        dst.write(src_image.astype(np.uint8), 1)

    # Run OTB command
    tmp_glcm = os.path.join(tmp_dir.name, 'glcm.tif')
    subprocess.run([
        'otbcli_HaralickTextureExtraction', '-in', tmp_image,
        '-parameters.xrad', str(x_radius), '-parameters.yrad', str(y_radius),
        '-parameters.xoff', str(x_offset), '-parameters.yoff', str(y_offset),
        '-parameters.min', str(image_min), '-parameters.max', str(image_max),
        '-parameters.nbbin', str(nb_bins), '-texture', kind,
        '-out', tmp_glcm, 'double'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Save each texture in an individual GeoTIFF
    os.makedirs(dst_dir, exist_ok=True)
    with rasterio.open(tmp_glcm) as src:

        for i, texture in enumerate(TEXTURES[kind]):
            
            profile = src.profile
            img = src.read(i+1).astype(np.float64)

            # Linear rescale and convert to UINT16
            img = np.interp(img, (img.min(), img.max()), (0, 65535))
            img = img.astype(np.uint16)
            profile.update(dtype=img.dtype.name)

            # Save as 1-band compressed GeoTIFF
            profile.update(compression='LZW', count=1, transform=None)
            filename = f'{prefix}{texture}_{x_radius*2+1}x{y_radius*2+1}.tif'
            output_file = os.path.join(dst_dir, filename)
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(img, 1)

    tmp_dir.cleanup()
    return
