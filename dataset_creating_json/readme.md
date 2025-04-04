# Data

*   Download the JSON file from Label Studio and rename it to `data.json`.

# Running the script on Metacentrum

1.  Create a new environment: `# mamba env create -f /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/conda-knn.yml --prefix /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/knn`
2.  Compile or download the Tesseract binary to the path `/storage/brno2/home/xvlkja07/local/bin/tesseract`. Alternatively, you can change the path in the `createDataset.py` file (or use the binary compiled by me).
3.  Run the script using `qrun runcomputing.sh`.

# Invalid Dates

*   Images that were not correctly processed by OCR are located in the `../img/problematic/` directory.



