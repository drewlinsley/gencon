#!/bin/bash

#SBATCH -p bigmem
#SBATCH -n 32
#SBATCH --mem=750G
#SBATCH -t 8:00:00
#SBATCH -o %j.output
#SBATCH -e %j.error

# # Set up the environment by loading modules
module load anaconda
source activate wk
cd /cifs/data/tserre_lrs/projects/prj_connectomics/gencon
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/postprocess/merge_segments.py configs/W-Q.yml
python src/postprocess/combine_merge_and_upload_full.py configs/W-Q.yml

