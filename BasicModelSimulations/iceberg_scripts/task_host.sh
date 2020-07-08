#!/bin/bash
#$ -cwd
#$ -V
#$ -l h_rt=00:01:00
#$ -l rmem=1G
host_list=(150 151)
for a in "${host_list[@]}"
do
  name="sharc-node$a"
  qsub -l hostname=$name task_job.sh
done

