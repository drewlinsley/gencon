#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH -n 20
#SBATCH --mem=80G
#SBATCH -J connectomics_reconstruction
#SBATCH -o package_ribbons.out
#SBATCH -e package_ribbons.err

module load anaconda
source activate wk
python package_ribbons.py

