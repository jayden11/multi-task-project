#!/bin/bash -l
# XXX: TODO: Script description.

echo 'Running Model'
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=60:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N lstm-gru
#$ -wd /home/jgodwin/
#$ -t 1-6
#$ -o ./data/outputs/grid_output/
#$ -e ./data/outputs/grid_output/

export PYTHONPATH=${PYTHONPATH}:/home/jgodwin/

timestamp=`date -u +%Y-%m-%dT%H%MZ`
directory="conll-grid"
mkdir "./data/outputs/${directory}"

i=$(expr $SGE_TASK_ID - 1)

num_steps=(64)
encoder_size=(400)
decoder_size=(400)
dropout=("0.5" "0.6" "0.7")
batch_size=(128)
embedding_size=(300)
num_shared_layers=(1)
num_private_layers=(1)
bidirectional=(1)
lstm=(1)
mix_percent=("0.3" "0.4")

total_steps=$((${#num_steps[@]} * ${#encoder_size[@]} * ${#decoder_size[@]} * ${#dropout[@]} * ${#batch_size[@]} * ${#embedding_size[@]} * ${#num_shared_layers[@]} * ${#num_private_layers[@]} * ${#bidirectional[@]} * ${#lstm[@]} ))
echo ${total_steps}
mix_percent_idx=$((i / total_steps))

i=$((i % total_steps))
steps=$(($total_steps / ${#lstm[@]}))
echo ${total_steps}
lstm_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#bidirectional[@]}))
echo ${steps}
bidirectional_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#num_private_layers[@]}))
echo ${steps}
num_private_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#num_shared_layers[@]}))
num_shared_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#embedding_size[@]}))
echo ${steps}
embedding_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#batch_size[@]}))
echo ${steps}
batch_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#dropout[@]}))
echo ${steps}
dropout_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#decoder_size[@]}))
echo ${steps}
decoder_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#encoder_size[@]}))
echo ${steps}
encoder_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#num_steps[@]}))
echo ${steps}
num_steps_idx=$((i / steps))

name="Conll_${num_steps[num_steps_idx]}n_${encoder_size[encoder_size_idx]}e_${decoder_size[decoder_size_idx]}d_${dropout[dropout_idx]}drop_${batch_size[batch_size_idx]}b_${embedding_size[embedding_size_idx]}emb_${num_shared_layers[num_shared_idx]}s_${num_private_layers[num_private_idx]}p_${bidirectional[bidirectional_idx]}bi_${lstm[lstm_idx]}cell_${mix_percent[mix_percent_idx]}mix"
echo ${name}

cp -R ./data/outputs/random_lm_150_July28_conll_annealing_glove_window_proj  ./data/outputs/${directory}/${name}
LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
  ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
  ./multi-task-project/lm/run_model.py --model_type "JOINT" \
    --dataset_path "./data/conll" \
    --ptb_path "./data/ptb" \
    --save_path "./data/outputs/conll-grid/${timestamp}" \
    --glove_path "./data/glove.6B/glove.6B.300d.txt"\
    --num_steps ${num_steps[num_steps_idx]} \
    --encoder_size ${encoder_size[encoder_size_idx]} \
    --pos_decoder_size ${encoder_size[encoder_size_idx]} \
    --chunk_decoder_size ${encoder_size[encoder_size_idx]} \
    --dropout ${dropout[dropout_idx]} \
    --batch_size ${batch_size[batch_size_idx]} \
    --pos_embedding_size ${embedding_size[embedding_size_idx]} \
    --num_shared_layers ${num_shared_layers[num_shared_idx]} \
    --num_private_layers ${num_private_layers[num_private_idx]} \
    --chunk_embedding_size ${embedding_size[embedding_size_idx]} \
    --lm_decoder_size  ${decoder_size[decoder_size_idx]} \
    --bidirectional ${bidirectional[bidirectional_idx]} \
    --lstm ${lstm[lstm_idx]} \
    --mix_percent ${mix_percent[mix_percent_idx]} \
    --write_to_file 0 \
    --embedding 1 \
    --max_epoch 30 \
    --test 0 \
    --projection_size 100 \
    --num_gold 15 \
    --reg_weight 0.01 \
    --word_embedding_size 300 \
    --embedding_trainable 0 \
    --adam 1 | tee "./data/outputs/${directory}/${name}.txt" \
