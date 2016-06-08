#!/bin/bash -l
# XXX: TODO: Script description.
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2016-02-25
echo 'Running Model'


#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=72:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N conll-batch-grid-search
#$ -wd /home/jgodwin/
export PYTHONPATH=${PYTHONPATH}:/home/jgodwin/

timestamp=`date -u +%Y-%m-%dT%H%MZ`
mkdir "./data/outputs/${timestamp}"
pids=''
for num_steps in 32 64 128
do
  #for depth in 1 2 3 4
  for encoder_size in 128 256
  do
    for decoder_size in 256 128
    do
      for dropout in 0.5 0.7 0.9
      do
        for batch_size in 128 64
        do
          for embedding_size in 400 600 800
          do
            for num_shared_layers in 1 2
            do
              for num_private_layers in 1 2
              do
                for bidirectional in "True" "False"
                do
                  for lstm in "True" "False"
                  do
                    for mix_percent in 0.1 0.2 0.3 0.4 0.5
                    do
                    name="Conll_${num_steps}n_${encoder_size}e_${decoder_size}d_${dropout}drop_${batch_size}b_${embedding_size}emb_${num_shared_layers}s_${num_private_layers}p_${bidirectional}bi_${lstm}cell"
                    LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64' \
                      ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
                      ./multi-task-project/lm/run_model.py --model_type "JOINT" \
                                           --dataset_path "./data/conll" \
                                           --ptb_path "./data/ptb" \
                                           --save_path "./data/outputs/${timestamp}" \
                                           --num_steps ${num_steps} \
                                           --encoder_size ${encoder_size} \
                                           --pos_decoder_size ${decoder_size} \
                                           --chunk_decoder_size ${decoder_size} \
                                           --dropout ${dropout} \
                                           --batch_size ${batch_size} \
                                           --pos_embedding_size ${embedding_size} \
                                           --num_shared_layers ${num_shared_layers} \
                                           --num_private_layers ${num_private_layers} \
                                           --chunk_embedding_size ${embedding_size} \
                                           --lm_decoder_size  ${decoder_size} \
                                           --bidirectional ${bidirectional} \
                                           --lstm ${lstm} \
                                           --mix_percent ${mix_percent} \
                                           --write_to_file "False" -o "./data/outputs/${timestamp}/${name}.txt"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
