#!/bin/bash
#$ -V

node_array=( 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
       146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
       159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
       172, 173, 174, 175, 176, 177)


for i in $(seq 1 200): 
do
	let val=$(($i%56))
	name="sharc-node${node_array[val]}"
	echo submitting job to $name
	qsub -l hostname=$name directed_node_job.sh $i 
	#echo ${node_array[val]}
done








