#!/bin/bash
#PBS -N KNN-DONUT-TRAINING
#PBS -l walltime=05:00:00
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=9:ngpus=1:mem=128gb:gpu_mem=43gb:scratch_ssd=80000mb

# Creating a unique directory on scratch
# SCRATCH_DIR=/scratch.ssd/xnevar00/job_${PBS_JOBID}
# echo "SCRATCH_DIR: ${SCRATCH_DIR}"
# mkdir -p ${SCRATCH_DIR}

# Copy only the 'donut' folder and the dataset file from 'dataset_creating_json' using rsync
rsync -avP /auto/brno2/home/xnevar00/donut ${SCRATCHDIR}/
rsync -avzP /auto/brno2/home/xnevar00/dataset/dataset.tar.gz ${SCRATCHDIR}/donut

cd ${SCRATCHDIR}/donut/

echo "Extracting dataset.tar.gz"
pigz -dc dataset.tar.gz | tar xf -

# Listing folder contents
# ls -laR ${SCRATCH_DIR}

# Initializing environment
# source /storage/brno2/home/xvlkja07/.bashrc
module add mambaforge

# Creating a new conda environment from the YAML file in the scratch folder
# mamba env create -f /storage/brno2/home/xvlkja07/KNN/donut_training/conda-knn-donut.yml --prefix ${SCRATCH_DIR}/knn-donut

# Activating the newly created environment
mamba activate /auto/brno2/home/xnevar00/donut/new-donut

# Setting the working directory – we will operate from the 'donut' folder
cd ${SCRATCHDIR}/donut

# Define source and destination for results copying
RESULT_SOURCE=${SCRATCHDIR}/donut/result/
RESULT_DEST=/auto/brno2/home/xnevar00/training_result

# Logging job start information
echo "$PBS_JOBID is running on node $(hostname -f)" >> /auto/brno2/home/xnevar00/donut/jobs_info.txt

echo "Model will be saved to $(pwd)"
echo "Starting training from the scratch environment"


# copy old results

# Start a background process that copies the results every 10 minutes
(
  while true; do
    echo "Periodically copying results from ${RESULT_SOURCE} to ${RESULT_DEST}..."
    mkdir -p $(dirname ${RESULT_DEST})
    rsync -avP ${RESULT_SOURCE} ${RESULT_DEST}
    sleep 600  # Wait for 10 minutes
  done
) &
COPY_PID=$!

echo "Starting train.py"

# Running training
python train2.py --config "config/train_cord.yaml" \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base-finetuned-cord-v2" \
                --dataset_name_or_paths "['./dataset']" \
                --exp_version "test_experiment"



# After training, kill the periodic copying process
echo "Training finished. Stopping periodic results copying process (PID: ${COPY_PID})."
kill $COPY_PID

# Optionally, do one final copy of the results
echo "Final copy of results from ${RESULT_SOURCE} to ${RESULT_DEST}"
mkdir -p $(dirname ${RESULT_DEST})
rsync -av ${RESULT_SOURCE} ${RESULT_DEST}
