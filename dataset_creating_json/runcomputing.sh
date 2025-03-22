#!/bin/bash
#PBS -N OCR-KNN
#PBS -l walltime=1:0:0
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=32:mem=32gb:scratch_local=400mb:arch=linux

# source conda init
source /storage/brno2/home/xvlkja07/.bashrc

# mamba env create -f /storage/brno2/home/xvlkja07/KNN/conda-knn.yml --prefix /storage/brno2/home/xvlkja07/KNN/knn

# module add tesseract
module add mambaforge
# mamba env create -f /storage/brno2/home/xvlkja07/KNN/conda-knn.yml
mamba activate /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/knn

HOMEDIR=/storage/brno2/home/xvlkja07/dataset_creating_json/KNN/

echo "$PBS_JOBID is running on node `hostname -f`" >> $HOMEDIR/jobs_info.txt

cd $HOMEDIR

echo "Model will be saved to $SAVE_MODEL_PATH"
echo "Starting OCR"

# Construct the command with the parameters
python createDataset.py