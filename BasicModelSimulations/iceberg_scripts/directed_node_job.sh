#!/bin/bash
#$ -l rmem=4G
#$ -l h_rt=00:32:00
#$ -o /home/$USER/logs/output
#$ -e /home/$USER/logs/error


echo "TASK ID is $1"

module load apps/matlab/2016b/binary

matlab -nodisplay -r "TCmodel_arrayjob_big_parallel($i, 960)"
