#!/bin/bash

echo How many jobs do you want to launch?
read num_jobs

for (( i=0; i<$num_jobs; i++ ))
do
    sbatch scripts/slurm_job.sh
done


