#!/bin/bash
#PBS -N KNN-DONUT-TRAINING
#PBS -l walltime=02:10:00
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=9:ngpus=1:mem=128gb:gpu_mem=43gb:scratch_ssd=400000mb

# Creating a unique directory on scratch
SCRATCH_DIR=/scratch.ssd/xvlkja07/job_${PBS_JOBID}
echo "SCRATCH_DIR: ${SCRATCH_DIR}"
# mkdir -p ${SCRATCH_DIR}

# Copy only the 'donut' folder and the dataset file from 'dataset_creating_json' using rsync
rsync -avP /storage/brno2/home/xvlkja07/KNN/donut ${SCRATCH_DIR}/
rsync -avzP /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/dataset.tar.gz ${SCRATCH_DIR}/donut

cd ${SCRATCH_DIR}/donut/

echo "Extracting dataset.tar.gz"
pigz -dc dataset.tar.gz | tar xf -

# Listing folder contents
ls -laR ${SCRATCH_DIR}

# Initializing environment
# source /storage/brno2/home/xvlkja07/.bashrc
module add mambaforge

# Creating a new conda environment from the YAML file in the scratch folder
# mamba env create -f /storage/brno2/home/xvlkja07/KNN/donut_training/conda-knn-donut.yml --prefix ${SCRATCH_DIR}/knn-donut

# Activating the newly created environment
mamba activate /storage/brno2/home/xvlkja07/KNN/donut_training/knn-donut

# Setting the working directory – we will operate from the 'donut' folder
cd ${SCRATCH_DIR}/donut

# Define source and destination for results copying
RESULT_SOURCE=${SCRATCH_DIR}/donut/result/
RESULT_DEST=/storage/brno2/home/xvlkja07/KNN/donut_training/result/

# Logging job start information
echo "$PBS_JOBID is running on node $(hostname -f)" >> /storage/brno2/home/xvlkja07/KNN/donut_training/jobs_info.txt

echo "Model will be saved to $(pwd)"
echo "Starting training from the scratch environment"

# Start a background process that copies the results every 10 minutes
(
  while true; do
    echo "Periodically copying results from ${RESULT_SOURCE} to ${RESULT_DEST}..."
    mkdir -p $(dirname ${RESULT_DEST})
    rsync -av ${RESULT_SOURCE} ${RESULT_DEST}
    sleep 600  # Wait for 10 minutes
  done
) &
COPY_PID=$!

echo "Starting train.py --config 'config/train_cord.yaml' \
                --pretrained_model_name_or_path naver-clova-ix/donut-base \
                --dataset_name_or_paths ['./dataset/'] \
                --exp_version test_experiment"
# Running training
python train.py --config "config/train_cord.yaml" \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base" \
                --dataset_name_or_paths "['./dataset/']" \
                --exp_version "test_experiment"



# After training, kill the periodic copying process
echo "Training finished. Stopping periodic results copying process (PID: ${COPY_PID})."
kill $COPY_PID

# Optionally, do one final copy of the results
echo "Final copy of results from ${RESULT_SOURCE} to ${RESULT_DEST}"
mkdir -p $(dirname ${RESULT_DEST})
rsync -av ${RESULT_SOURCE} ${RESULT_DEST}