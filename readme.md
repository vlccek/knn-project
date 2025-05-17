# Project KNN - sementic abstraction of headings

*Advanced course on BUT FIT about NN and ML.* 

This repository provides a comprehensive pipeline for creating datasets, training Document Understanding Transformer (Donut) models, and deploying them for inference. The project is organized into several key directories, each addressing a distinct aspect of the Donut workflow and the LayoutLMv3 implementation with same approach.

## Directory Structure

*   **`img/`:** (Image Assets) This directory contains visual resources used within this `README.md` or in other parts of the project.

*   **`dataset generation/`:** Contains resources used in dataset generation process
    *   **Overview:** The heart of this project lies within its capacity to create high-quality datasets tailored for document understanding. The folder contain various tools and configurations
    * the Dataset create by LLM can be downloaded from [here](GT_jsons are available at [this repo](https://github.com/vlccek/knn-responses))

*   **`donut/`:** This Directory contain core model of donut which is a Document Understanding Transformer models and our extension of it. 

*   **`donut_training/`:** The folder will contain training scripts for training on metacentrum and env. 

## **`pero_layout`** 

Folder containing all data and source codes for our PERO + LayoutLMv3 solution.
*  `datasets` - contains datasets that were used for training (without images, since they are too big) along with the process for their creation.

    * `convert_images.py` - converts all images to RGB format
    * `create_dataset.py` - creates 3 dataset files, one for each split (80:10:10 ratio)
    * `annotations.json` - image annotations downloaded from LabelStudio
    * `created_datasets` folder containing datasets for train, test and validation splits.

*  `training` - folder containing training script used for LayoutLMv3 finetuning on Google Colab.

*  `pipeline` - folder containing whole final processing pipeline.
    * `images` - images to be processed
    * `processing` - folder containing `PeroProcessor.py` for processing PERO OCR outputs, `LayoutProcessor.py` for Layout inference and `ocr-to-json.py` controling the whole process and converting the outputs of LayoutLMv3 to JSON.
    *  `requirements.txt`
    *  `run.sh` - script for launching the processing of images in `images` folder


## How to run PERO + LAYOUTLMv3 inference

1. Download configured Layout and PERO models from [this url](https://drive.google.com/drive/folders/1TLxB4ENP5-d_-lFLbSLXGB39YVsoovVX?usp=sharing) (`models.tar.gz`) and place the TAR in the `pipeline` folder (you should generate 2 new folders).

2. `tar -xzvf models.tar.gz` 
3. Make sure you have desired images in `images` folder to be processed. Image can be downloaded from merlin.
4. `./run.sh`, the output folder is set to `./output/structure/` along with the OCR outputs.

