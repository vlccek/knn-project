# qsub -I -l select=1:ncpus=8:ngpus=1:mem=128gb:gpu_mem=16gb:scratch_local=40000mb -l walltime=00:30:00


SCRATCH_DIR=/scratch.ssd/xvlkja07/job_${PBS_JOBID}
echo "SCRATCH_DIR: ${SCRATCH_DIR}"

rsync -avP /storage/brno2/home/xvlkja07/KNN/donut ${SCRATCH_DIR}/
rsync -avzP /storage/brno2/home/xvlkja07/KNN/dataset_creating_json/dataset.tar.gz ${SCRATCH_DIR}/donut

cd ${SCRATCH_DIR}/donut/

echo "Extracting dataset.tar.gz"
pigz -dc dataset.tar.gz | tar xf -


module add mambaforge

mamba activate /storage/brno2/home/xvlkja07/KNN/donut_training/knn-donut

# Setting the working directory – we will operate from the 'donut' folder
cd ${SCRATCH_DIR}/donut

python test.py --dataset_name_or_path './dataset/' --pretrained_model_name_or_path /storage/brno2/home/xvlkja07/KNN/donut_training/result/train_cord/test_experiment --save_path ./result/output.json

# python test.py --dataset_name_or_path '/storage/brno2/home/xvlkja07/KNN/donut/dataset' --pretrained_model_name_or_path /storage/brno2/home/xvlkja07/KNN/donut_training/result2/train_cord/test_experiment --save_path ./result/output.json
