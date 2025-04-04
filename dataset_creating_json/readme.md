# Folder Files Overview

This directory contains the following files, related to dataset creating for KNN project.

* **`conda-knn.yml`:** A Conda environment file. This file lists the dependencies required to run the KNN portion of the
  project and create a Conda virtual environment. Used to test the tesseract OCR.

* **`createDataset.py`:** A Python script responsible for creating a dataset. File for creating datataset by using
  tesseract OCR.

* **`createDatasetDonutFormat.py`:** Takes the dataset created by `createDatasetGemmini.py` and converts this to DONUT
  format dataset. Ouput in dataset folder

* **`createDatasetGemmini.py`:** From pictures in ../img/ creates ./response/ folder by gemini API.

* **`runcomputing.sh`:** A shell script that runs the `createDataset.py` script on metacentrum.

* **`verifyDataset.py`:** A Python script for verifying the integrity of the dataset. This might involve checking for
  missing values, data consistency, or distribution characteristics.