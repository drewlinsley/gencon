#!/bin/bash

#SBATCH -p gpu-he --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=110G
#SBATCH -t 8:00:00
#SBATCH -o %j.output
#SBATCH -e %j.error

# # Set up the environment by loading modules
module load anaconda
module load cudnn
module load cuda/11.1.1 gcc/10.2
source activate wk
cd /cifs/data/tserre_lrs/projects/prj_connectomics/gencon
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/process/segment_routine.py configs/somatosensory.yml

