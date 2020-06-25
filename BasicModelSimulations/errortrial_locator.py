#!/usr/bin/python

import os
import numpy as np
from natsort import natsorted
import pathlib


error_folder = input("Path to error files: ")

sge_folder = input("Specify name of folder where SGE files will be saved: ")
folder_directory = os.path.join(os.getcwd(), sge_folder)

if not os.path.exists(folder_directory):
    os.mkdir(folder_directory)

os.chdir(folder_directory)

for i,file in enumerate (natsorted(os.listdir(error_folder))):
    file_dir = os.path.join(error_folder, file)
    
    if (os.stat(file_dir).st_size > 0):
        error_experiment = pathlib.Path(file_dir).suffix[1:]
        print(error_experiment)
        
        f = open("error_redo" + error_experiment + ".sge", "x")
        f.write("#!/bin/bash\n")
        f.write("#$ -N Arrayjobproject\n")
        f.write("#$ -l rmem=4G\n")
        f.write("#$ -o /home/$USER/logs/output\n")
        f.write("#$ -e /home/$USER/logs/error\n")
        f.write("#$ -t "+ error_experiment+"\n")
        f.write("module load apps/matlab/2016b/binary\n")
        f.write("matlab -nodisplay -r \"TCmodel_arrayjob($SGE_TASK_ID, 9600)\"")
        f.close()