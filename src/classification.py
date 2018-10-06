"""Perform classifications using landsat, sentinel-1 or both."""

import os

import rasterio
import rasterio.features
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from scipy.ndimage import uniform_filter

from metadata import CASE_STUDIES, DATA_DIR


def list_available_features(data_dir):
    """List all available images in a given directory and its
    subdirectories.

    Parameters
    ----------
    data_dir : str
        Path to the directory where images are stored.

    Returns
    -------
    features : list of tuple
        Available features as a list of tuples (label, path).
    """
    features = []
    for directory, _, files in os.walk(data_dir):
        files = [f for f in files if f.endswith('.tif')]
        for f in files:
            path = os.path.join(directory, f)
            label = f.replace('.tif', '')
            features.append((label, path))
    return features


def ndarray_from_images(images, mask):
    """Construct X ndarray from an iterable of images and according to
    a provided binary raster mask.

    Parameters
    ----------
    images : iterable of numpy 2d arrays
        Images as an iterable of numpy 2d arrays.
    mask : binary numpy 2d array
        Raster mask ; true pixels will be excluded.

    Returns
    -------
    X : numpy array
        Array of shape (n_samples, n_images).
    """
    # Initialize the X array of shape (n_samples, n_images)
    out_shape = (images[0][~mask].ravel().shape[0], len(images))
    X = np.empty(shape=out_shape, dtype=np.float64)

    # Populate with image data
    for i, img in enumerate(images):
        X[:, i] = img[~mask].ravel()
    return X


def get_train_test(data_dir, case_study, test_size=0.3, seed=111):
    """Construct train and test rasters from reference land cover shapefiles.
    Train and test samples are randomly splitted at the polygon-level.

    Parameters
    ----------
    data_dir : str
        Path to the directory where reference shapefiles are stored.
    case_study : Metadata
        Metadata object corresponding to a given case study. Used
        to retrieve rasterization parameters.
    test_size : float
        Size of the test sample (between 0 and 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train : 2d array
        Training samples as a 2d numpy array.
    test : 2d array
        Testing samples as a 2d numpy array.
    """
    # Get built-up train and test samples
    bu = gpd.read_file(os.path.join(reference_dir, 'builtup.shp'))
    if bu.crs != case_study.crs:
        bu = bu.to_crs(case_study.crs)
    bu_train, bu_test = train_test_split(
        bu, test_size=test_size, random_state=seed)
    train = rasterio.features.rasterize(
        shapes=((geom, 1) for geom in bu_train.geometry),
        out_shape=(case_study.height, case_study.width),
        transform=case_study.affine, dtype=np.uint8)
    test = rasterio.features.rasterize(
        shapes=((geom, 1) for geom in bu_test.geometry),
        out_shape=(case_study.height, case_study.width),
        transform=case_study.affine, dtype=np.uint8)

    # Get non-built-up train and test samples
    # Each land cover is splitted individually at the polygon-level
    # Legend: so=2, lv=3, hv=4
    NB_LAND_COVERS = ['baresoil', 'lowveg', 'highveg']
    for i, land_cover in enumerate(NB_LAND_COVERS):
        lc = gpd.read_file(os.path.join(reference_dir, land_cover + '.shp'))
        if lc.crs != case_study.crs:
            lc = lc.to_crs(case_study.crs)
        lc_train, lc_test = train_test_split(
            lc, test_size=test_size, random_state=seed)
        lc_train_raster = rasterio.features.rasterize(
            shapes=((geom, 1) for geom in lc_train.geometry),
            out_shape=(case_study.height, case_study.width),
            transform=case_study.affine, dtype=np.uint8)
        lc_test_raster = rasterio.features.rasterize(
            shapes=((geom, 1) for geom in lc_test.geometry),
            out_shape=(case_study.height, case_study.width),
            transform=case_study.affine, dtype=np.uint8)
        train[lc_train_raster == 1] = i + 2
        test[lc_test_raster == 1] = i + 2

    return train, test


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


def random_forest(X, train, mask, seed=111, **kwargs):
    """Classify image data based on a given training dataset with
    the Random Forest classifier.

    Parameters
    ----------
    X : array
        Image data to classify. Array of shape
        (n_samples, n_features).
    train : 2d array
        Training samples as a 2d numpy array.
    mask : 2d array
        Pixels to exclude from the analysis.
    seed : int
        Random seed for reproducibility.
    **kwargs : args
        Parameters to pass to the RF classifier.

    Returns
    -------
    probabilities : 2d array
        RF probabilistic output as a 2d image.
    importances : array
        RF feature importances as an array of shape
        (n_features).
    """
    # Construct training dataset from X based on `train`
    X_train = X[train[~mask].ravel() > 0, :]
    y_train = train[~mask].ravel()[train[~mask].ravel() > 0]
    y_train[y_train > 1] = 2

    # Oversampling to handle class imbalance
    ros = RandomOverSampler(random_state=seed)
    X_train, y_train = ros.fit_sample(X_train, y_train)

    # Classification with RF
    rf = RandomForestClassifier(random_state=seed, **kwargs)
    rf.fit(X_train, y_train)
    X_pred = rf.predict_proba(X)[:, 0]

    # Reconstruct probabilities raster with original shape
    probabilities = np.zeros(shape=mask.shape, dtype=np.float64)
    probabilities[~mask] = X_pred

    return probabilities, rf.feature_importances_


def assess(probabilities, test, mask, prob_threshold=0.75):
    """Compute assessment metrics based on the provided test samples.
    Metrics computed: F1-score, precision, recall, and accuracy
    in each land cover.

    Parameters
    ----------
    probabilities : 2d array
        RF probabilistic output.
    test : 2d array
        Test samples.
    mask : 2d array
        Pixels excluded from the analysis.
    prob_threshold : float
        RF probabilities binary threshold.

    Returns
    -------
    metrics : pandas Serie
        Assessment metrics.
    """
    metrics = pd.Series()
    test[mask] = 0
    y_pred = probabilities[test > 0].ravel()
    y_true = test[test > 0].ravel()
    y_pred = y_pred >= prob_threshold
    y_true = y_true == 1
    metrics.at['f1_score'] = f1_score(y_true, y_pred)
    metrics.at['precision'] = precision_score(y_true, y_pred)
    metrics.at['recall'] = recall_score(y_true, y_pred)

    # Accuracy in each land cover
    land_cover_names = ['builtup', 'baresoil', 'lowveg', 'highveg']
    land_cover_values = [1, 2, 3, 4]
    for name, value in zip(land_cover_names, land_cover_values):
        y_pred = probabilities[test == value].ravel()
        y_true = test[test == value].ravel()
        y_pred = y_pred >= prob_threshold
        y_true = y_true == 1
        metrics.at[f'{name}_accuracy'] = accuracy_score(y_true, y_pred)

    return metrics


def post_process(probabilities, kernel_size=3):
    """Post-process RF probabilistic output with a 2d uniform filter."""
    return uniform_filter(probabilities, size=kernel_size)


def scheme_computed(case_study, scheme_label):
    """Check if a given scheme has already been computed."""
    expected_dir = os.path.join(DATA_DIR, 'output', case_study.id, scheme_label)
    if not os.path.isdir(expected_dir):
        return False
    return os.path.isfile(os.path.join(expected_dir, 'metrics.csv'))


if __name__ == '__main__':

    # Size of the test sample (between 0 and 1) during train/test split
    # of the reference polygons.
    TEST_SIZE = 0.5

    # Fixed random seed for reproducibility
    SEED = 111

    # Number of classifications with different random train/test splits
    N = 10

    # Assessment metrics to compute
    METRICS = [
        'f1_score', 'precision', 'recall', 'builtup_accuracy',
        'baresoil_accuracy', 'lowveg_accuracy', 'highveg_accuracy'
    ]

    LANDSAT_BANDS = [
        'blue', 'green', 'red', 'nir',
        'swir1', 'swir2', 'tir1', 'tir2'
    ]

    # Eight different classification schemes with various
    # input data :
    #   1. Landsat bands (ndim=8)
    #   2. PCA(VV 5x5) (ndim=6)
    #   3. PCA(VH 5x5) (ndim=6)
    #   4. PCA(VV 5x5) + PCA(VH 5x5) (ndim=12)
    #   5. PCA(VV 7x7) + PCA(VH 7x7) (ndim=12)
    #   6. PCA(VV 9x9) + PCA(VH 9x9) (ndim=12)
    #   7. PCA(VV 5x5) + PCA(VH 5x5) +
    #      PCA(VV 9x9) + PCA(VH 9x9) (ndim=24)
    #   8. Landsat bands + PCA(VV 5x5) + PCA(VH 5x5) (ndim=20)
    SCHEMES = [
        LANDSAT_BANDS,
        ['pca_vv_5x5'],
        ['pca_vh_5x5'],
        ['pca_vv_5x5', 'pca_vh_5x5'],
        ['pca_vv_7x7', 'pca_vh_7x7'],
        ['pca_vv_9x9', 'pca_vh_9x9'],
        ['pca_vv_11x11', 'pca_vh_11x11'],
        ['pca_vv_5x5', 'pca_vh_5x5', 'pca_vv_9x9', 'pca_vh_9x9'],
        ['pca_vv_5x5', 'pca_vh_5x5', 'pca_vv_11x11', 'pca_vh_11x11'],
        ['pca_vv_5x5', 'pca_vh_5x5'] + LANDSAT_BANDS,
        ['pca_vv_9x9', 'pca_vh_9x9'] + LANDSAT_BANDS,
        ['pca_vv_11x11', 'pca_vh_11x11'] + LANDSAT_BANDS
    ]

    SCHEMES_LABELS = [
        'optical',
        'sar_vv_5x5',
        'sar_vh_5x5',
        'sar_vv_vh_5x5',
        'sar_vv_vh_7x7',
        'sar_vv_vh_9x9',
        'sar_vv_vh_11x11',
        'sar_vv_vh_5x5_9x9',
        'sar_vv_vh_5x5_11x11',
        'optical_sar',
        'optical_sar_9x9',
        'optical_sar_11x11'
    ]

    progress = tqdm(total=len(CASE_STUDIES) * N * len(SCHEMES))

    for case_study in CASE_STUDIES:

        reference_dir = os.path.join(
            DATA_DIR, 'raw', 'reference', case_study.id)
        opt_dir = os.path.join(DATA_DIR, 'processed', 'landsat', case_study.id)
        sar_dir = os.path.join(DATA_DIR, 'processed',
                               'sentinel-1', case_study.id)
        opt_features = list_available_features(opt_dir)
        sar_features = list_available_features(sar_dir)
        available_features = opt_features + sar_features

        mask = case_study.water

        for feature_labels, scheme_label in zip(SCHEMES, SCHEMES_LABELS):

            # Abort if scheme already computed
            if scheme_computed(case_study, scheme_label):
                progress.update(N)
                continue

            features = [
                (label, path) for label, path in available_features
                if label in feature_labels
            ]

            images, labels = [], []
            for label, path in features:
                with rasterio.open(path) as src:
                    if 'pca' in label:
                        for i in range(6):
                            images.append(src.read(i+1))
                            labels.append(label + f'_c{i+1}')
                    else:
                        images.append(src.read(1))
                        labels.append(label)

            X = ndarray_from_images(images, mask)

            probabilities = np.zeros(shape=mask.shape, dtype=np.float64)
            metrics = pd.DataFrame(index=METRICS)
            importances = pd.DataFrame(index=labels)

            for i in range(N):

                train, test = get_train_test(
                    reference_dir, case_study, test_size=TEST_SIZE, seed=SEED+i
                )

                probabilities_, importances_ = random_forest(
                    X, train, mask, seed=SEED+i, n_estimators=50, n_jobs=6
                )

                probabilities_ = post_process(probabilities_, kernel_size=3)

                metrics_ = assess(probabilities_, test,
                                  mask, prob_threshold=0.75)
                metrics[i+1] = metrics_
                importances[i+1] = importances_
                probabilities += probabilities_

                progress.update(1)

            probabilities = probabilities / N

            output_dir = os.path.join(
                DATA_DIR, 'output', case_study.id, scheme_label
            )
            os.makedirs(output_dir, exist_ok=True)
            profile = case_study.profile.copy()
            profile.update(dtype=np.float64, transform=None)
            os.chdir(output_dir)
            with rasterio.open('probabilities.tif', 'w', **profile) as dst:
                dst.write(probabilities, 1)
            metrics.to_csv('metrics.csv')
            importances.to_csv('features_importances.csv')

    progress.close()
