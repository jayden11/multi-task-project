#!/bin/bash -l

echo 'Running Model'
#test

#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=12:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N multi-task
#$ -wd /home/jgodwin/

export PYTHONPATH=${PYTHONPATH}:/home/jgodwin/
LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
                ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
                ./multi-task-project/shared_model/run_model.py --model_type "JOINT" --dataset_path "./data/conll"
