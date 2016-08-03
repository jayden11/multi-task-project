#!/bin/bash -l

echo 'Running Model'


#$ -l tmem=12G
#$ -l h_vmem=12G
#$ -l h_rt=60:00:00
#These are optional flags but you problably want them in all jobs

#$ -l gpu=1,gpu_k40=1
#$ -S /bin/bash
#$ -N co-bi-lm-70-gpu
#$ -wd /home/jgodwin/
echo 'Run Model'
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
                ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
                ./multi-task-project/lm/run_model.py --model_type "JOINT" \
                     --dataset_path "./data/conll" \
                     --ptb_path "./data/ptb" \
                     --save_path "./data/outputs/random_lm_150_July28_conll_annealing_glove_window_proj/" \
		     --glove_path "./data/glove.6B/glove.6B.300d.txt" \
                     --num_steps 128 \
                     --encoder_size 256 \
                     --pos_decoder_size 256 \
                     --chunk_decoder_size 256 \
                     --dropout 0.5 \
                     --batch_size 128 \
                     --pos_embedding_size 50 \
                     --num_shared_layers 1 \
                     --num_private_layers 1 \
                     --chunk_embedding_size 50 \
                     --lm_decoder_size  256 \
                     --bidirectional 1 \
                     --lstm 0 \
                     --mix_percent 0.4 \
                     --write_to_file 1 \
                     --embedding 1 \
		     --max_epoch 300 \
                     --test 0 \
		     --num_gold 5 \
                     --reg_weight 1e-10 \
		     --word_embedding_size 300 \
		     --projection_size 300 \
		     --adam 1 \
		     --embedding_trainable 1 \
