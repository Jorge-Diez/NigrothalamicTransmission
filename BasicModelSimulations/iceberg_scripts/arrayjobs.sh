#!/bin/bash
#$ -N Arrayjobproject
#$ -l rmem=4G
#$ -M jdiezgarcia-victoria1@sheffield.ac.uk
#$ -m bea
#$ -o /home/$USER/logs
#$ -e /home/$USER/logs

#$ -t 1-8


echo "TASK ID is $SGE_TASK_ID"

module load apps/matlab/2016b

matlab -nodisplay -r "TCmodel_arrayjob($SGE_TASK_ID, 8)"

