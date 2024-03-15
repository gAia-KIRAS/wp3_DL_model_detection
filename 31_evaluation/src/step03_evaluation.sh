#!/bin/bash

#SBATCH --job-name=landslide
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
#SBATCH --gres=gpu:2080:1
source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

conda activate ls02

export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"

cp ./../../12_output_csv/*csv ./../data/operation_records/results_cd.csv
python main.py

