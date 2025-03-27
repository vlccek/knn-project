#!/bin/bash
#PBS -N KNN-DONUT-TRAINING
#PBS -l walltime=00:30:0
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=4:ngpus=1:mem=512gb:gpu_mem=16gb:scratch_local=400mb


# Vytvoření unikátního adresáře na scratch
SCRATCH_DIR=/scratch.ssd/xvlkja07/${PBS_JOBID}
# mkdir -p ${SCRATCH_DIR}

# Zkopírujeme pouze složky donut a dataset_creating_json pomocí rsync
rsync -avP /storage/brno2/home/xvlkja07/KNN/donut ${SCRATCH_DIR}/
rsync -avP /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/dataset ${SCRATCH_DIR}/

# Inicializace prostředí
# source /storage/brno2/home/xvlkja07/.bashrc
module add mambaforge

# Vytvoření nového conda prostředí z YAML souboru do složky na scratch
mamba env create -f /storage/brno2/home/xvlkja07/KNN/conda-knn-donut.yml --prefix ${SCRATCH_DIR}/knn-donut

# Aktivace nově vytvořeného prostředí
mamba activate ${SCRATCH_DIR}/knn-donut

# Nastavení pracovního adresáře – budeme pracovat ze složky donut
cd ${SCRATCH_DIR}/donut



# Zápis informace o spuštění do logu
echo "$PBS_JOBID is running on node $(hostname -f)" >> jobs_info.txt

echo "Model bude uložen do $(pwd)"
echo "Spouštím Trénování z prostředí na scratch"

# Spuštění tréninku
python train.py --config "config/train_cord.yaml" \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base" \
                --dataset_name_or_paths "${SCRATCH_DIR}/dataset/" \
                --exp_version "test_experiment"

# Po skončení tréninku zkopírujeme složku s výsledky zpět do úložiště pomocí rsync
RESULT_SOURCE=${SCRATCH_DIR}/donut/knn-donut/result
RESULT_DEST=/storage/brno2/home/xvlkja07/KNN/donut_training/knn-donut/result

echo "Kopíruji výsledky z ${RESULT_SOURCE} do ${RESULT_DEST}"
mkdir -p $(dirname ${RESULT_DEST})
rsync -av ${RESULT_SOURCE} ${RESULT_DEST}
