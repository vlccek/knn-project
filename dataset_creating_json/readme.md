# Project Overview: Donut Model Dataset Creation, Analysis, and KNN Integration

This repository is focused on creating, analyzing, and verifying datasets for use with Document Understanding Transformer (Donut) models.

## Files and Their Purpose

*   **`analyze_dataset_donut.ipynb`:** A Jupyter Notebook designed to analyze the structure and content of datasets generated for Donut models used for getting statistics and visualizations. 

*   **`conda-knn.yml`:** A Conda environment file. This file specifies the dependencies needed to create a Conda virtual environment.

*   **`createDataset.py`:** A Python script responsible for generating a dataset by infromation from anotations and OCR. 

*   **`createDatasetDonutFormat.py`:** Takes output from gemini API and creates a dataset in the format required by Donut models.

*   **`createDatasetGemmini.py`:** A Python script for creating dataset by gemini API.

*   **`readme.md`:** This file (the one you're reading now) provides an overview of the project, its files, and how to use them.

*   **`runcomputing.sh`:** A shell script for deploying the solution on metacentrum.

*   **`verifyDataset.py`:** A Python script used to verify the integrit generated datasets. This script might perform checks for missing values, data inconsistencies, and adherence to predefined schemas.

