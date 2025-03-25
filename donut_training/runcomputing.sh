#!/bin/bash
#PBS -N OCR-DONUT-TRAINING
#PBS -l walltime=4:0:0
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=32:ngpus=1:mem=512gb:gpu_mem=16gb:scratch_local=400mb

# source conda init
source /storage/brno2/home/xvlkja07/.bashrc

# mamba env create -f /storage/brno2/home/xvlkja07/KNN/conda-knn-donut.yml --prefix /storage/brno2/home/xvlkja07/KNN/knn

# module add tesseract
module add mambaforge
# mamba env create -f /storage/brno2/home/xvlkja07/KNN/conda-knn-donut.yml
mamba activate /storage/brno2/home/xvlkja07/KNN/donut_training/knn-donut

HOMEDIR=/storage/brno2/home/xvlkja07/KNN/donut_training/

echo "$PBS_JOBID is running on node `hostname -f`" >> $HOMEDIR/jobs_info.txt

cd $HOMEDIR

echo "Model will be saved to $HOMEDIR"
echo "Starting OCR"

# Construct the command with the parameters
python train.py