#!/bin/bash
#$ -V

node_array=( $(seq 120 195 ))


for i in $(seq 1 960): 
do
	let val=$(($i%76))
	qsub directed_node_job.sh $i -l hostname=sharc-node${node_array[val]}
	#echo ${node_array[val]}
done








