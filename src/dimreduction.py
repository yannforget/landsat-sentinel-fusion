"""Perform dimensionality reduction on SAR GLCM textures dataset
using PCA. Six PCA images with 6 bands (one per component) are created,
using different input data:

    1. GLCM VV 5x5 textures
    2. GLCM VH 5x5 textures
    3. GLCM VV 7x7 textures
    4. GLCM VH 7x7 textures
    5. GLCM VV 9x9 textures
    6. GLCM VH 9x9 textures

Output files are written in `{DATA_DIR}/processed/sentinel-1/{case_study}/pca`.
"""

import os
from itertools import product

import rasterio
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from metadata import CASE_STUDIES, DATA_DIR
from classification import list_available_features


def dim_reduction(X, n_components=6):
    """Perform dimensionality reduction on input data
    using PCA.

    Parameters
    ----------
    X : array
        Input data as an array of shape (n_samples, n_features).
    n_components : int
        PCA components.
    
    Returns
    -------
    X_reduced : array
        Output reduced data array of shape (n_samples, n_components).
    """
    pca = PCA(n_components=6)
    pca.fit(X)
    return pca.transform(X)


POLARISATIONS = ['vv', 'vh']
WINDOW_SIZES = ['5x5', '7x7', '9x9', '11x11']
N_COMPONENTS = 6

progress = tqdm(
    total=len(CASE_STUDIES) * len(POLARISATIONS) * len(WINDOW_SIZES)
)

for case_study in CASE_STUDIES:

    data_dir = os.path.join(
        DATA_DIR, 'processed', 'sentinel-1', case_study.id, 'textures')
    output_dir = os.path.join(data_dir, '..', 'pca')
    os.makedirs(output_dir, exist_ok=True)
    features = list_available_features(data_dir)
    
    for polarisation, window_size in product(POLARISATIONS, WINDOW_SIZES):

        # Abort if already computed
        output_fp = f'pca_{polarisation}_{window_size}.tif'
        if os.path.isfile(os.path.join(output_dir, output_fp)):
            progress.update(1)
            continue

        features_ = [
            feature for feature in features
            if polarisation in feature[0]
            and window_size in feature[0]
        ]

        # Create a X array of shape (n_samples, n_features)
        for label, path in features_:

            X = np.empty(
                shape=(case_study.width*case_study.height, len(features_)),
                dtype=np.uint16
            )

            for i, (label, path) in enumerate(features_):
                with rasterio.open(path) as src:
                    X[:, i] = src.read(1).ravel()

        # Perform dimensionality reduction and save result as a raster
        X = dim_reduction(X, n_components=N_COMPONENTS)
        X = X.reshape(
            case_study.height, case_study.width, N_COMPONENTS)
        os.chdir(output_dir)
        profile = case_study.profile.copy()
        profile.update(transform=None, dtype=X.dtype.name, count=N_COMPONENTS)
        with rasterio.open(output_fp, 'w', **profile) as dst:
            for i in range(N_COMPONENTS):
                dst.write(X[:, :, i], i+1)
        
        progress.update(1)

progress.close()
