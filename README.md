[![DOI](https://zenodo.org/badge/151975445.svg)](https://zenodo.org/badge/latestdoi/151975445)

This repository contains the Python code supporting the following paper:

* Forget Y., Shimoni M., Gilbert M., and Linard C. *Complementarity Between Sentinel-1 and Landsat 8 Imagery for Built-Up Mapping in Sub-Saharan Africa*, In Press, 2018.

Input and output datasets can be downloaded from [Zenodo](https://zenodo.org/record/1450932).

# Dependencies

Python dependencies are listed in the `environment.yml` and the `requirements.txt` files.

A virtual environment containing all the required dependencies can be automatically created using `conda`:

``` bash
# Clone the repository
git clone https://github.com/yannforget/landsat-sentinel-fusion.git
cd landsat-sentinel-fusion

# Create the virtual environment
conda env create --file environment.yml

# Activate the environment
source activate landsat-sentinel-fusion
```

The code also depends on:

* [Orfeo Toolbox](https://www.orfeo-toolbox.org/) for the computation of GLCM textures ;
* [SNAP](http://step.esa.int/main/download/) for SAR data preprocessing.

# Data

Input and output datasets are available in a [Zenodo deposit](https://zenodo.org/record/1450932).

``` bash
# Download and decompress the data
wget -O data.zip https://zenodo.org/record/1450932/files/data.zip?download=1
unzip data.zip
```

Validation samples can be found in `data/raw/reference` (as shapefiles) or in `data/processed/reference` (as rasters).

Classification outputs and performance metrics are located in `data/output` for each case study.

Due to storage constraints, input satellite imagery is not included in the Zenodo deposit. However, the product identifiers are available in `data/raw/landsat/products.txt` and `data/raw/sentinel-1/products.txt`. This means that they can be automatically downloaded using auxiliary software such as [landsatxplore](https://github.com/yannforget/landsatxplore) or [sentinelsat](https://github.com/sentinelsat/sentinelsat).

For Landsat 8 scenes:

``` bash
pip install landsatxplore

# Earth Explorer credentials
export LANDSATXPLORE_USERNAME=<your_username>
export LANDSATXPLORE_PASSWORD=<your_password>

cd data/raw/landsat

# Download each product with landsatxplore
for id in products.txt; do landsatxplore download $id; done

# Decompress each product
for product in *.zip; do unzip $product; done
```

For Sentinel-1 imagery:

``` bash
cd ../sentinel-1

# Install and configure sentinelsat
pip install sentinelsat
export DHUS_USER=<your_username>
export DHUS_PASSWORD=<your_password>

# Download Sentinel-1 products
for id in products.txt; do sentinelsat --download --name $id; done
```


# Code

## Running the analysis


``` bash
# Preprocessing of Optical and SAR data
python preprocess_landsat.py
python preprocess_sentinel1.py

# Dimensionality reduction (PCA) of SAR data
python dimreduction.py

# Random forest classification and validation
python classification.py
```

## Modules

* `src/glcm.py` : computing of GLCM textures using Orfeo Toolbox.
* `src/metadata.py` : accessing metadata specific to each case study.
* `src/raster.py` : various raster processing functions.
* `src/utils.py` : helper functions.

## Scripts

The following scripts has been used for the study but are not necessary to run the analysis :

* `src/aoi.py` : generates areas of interest for each case study.
* `src/climate.py` : monthly ndvi and precipitations for each case study.
* `src/land_masks.py` : land/water masks using openstreetmap data.
* `src/preprocess_reference.py` : rasterizes reference samples (polygons).