#!/bin/bash -l
# XXX: TODO: Script description.
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2016-02-25

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
                        name="Conll_${num_steps}n_${encoder_size}e_${decoder_size}d_${dropout}drop_${batch_size}b_${embedding_size}emb_${num_shared_layers}s_${num_private_layers}p_${bidirectional}bi_${lstm}cell"
                        echo name
                        -o "./data/outputs/${timestamp}/${name}.txt""
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
done
