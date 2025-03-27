#!/bin/bash
#PBS -N KNN-DONUT-TRAINING
#PBS -l walltime=4:0:0
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=4:ngpus=1:mem=128gb:gpu_mem=16gb:scratch_local=400mb

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
python train.py --config "../donut/config/train_cord.yaml" \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base" \
                --dataset_name_or_paths "[̈́'../dataset_creating_json/dataset/']" \
                --exp_version "test_experiment"