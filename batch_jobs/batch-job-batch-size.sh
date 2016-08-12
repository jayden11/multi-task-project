#!/bin/bash -l
# XXX: TODO: Script description.

echo 'Running Model'
#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -l h_rt=100:00:00
#These are optional flags but you problably want them in all jobs

#$ -S /bin/bash
#$ -N batch-size-batch
#$ -wd /home/jgodwin/
#$ -t 1-18
#$ -o ./data/outputs/grid_output/
#$ -e ./data/outputs/grid_output/

export PYTHONPATH=${PYTHONPATH}:/home/jgodwin/

timestamp="date -u +%Y-%m-%dT%H%MZ"
directory="batch-grid"
mkdir -p "./data/outputs/${directory}"


i=$(expr $SGE_TASK_ID - 1)

num_steps=(16 32 64)
encoder_size=(256)
decoder_size=(256)
dropout=("0.5")
batch_size=(16 32 64)
embedding_size=(300)
task_embedding_size=(50)
num_shared_layers=(1)
num_private_layers=(1)
bidirectional=(1)
lstm=(1)
mix_percent=(0.5)
embedding_path=("./data/glove.6B/glove.6B.300d.txt")
embedding=("glove")
dataset_path=("./data/conll" "./data/genia")
dataset=("Conll" "Genia")
ptb_path=("./data/ptb" "./data/pubmed")
projection_size=(100)
num_gold=(0)
reg_weight=("1e-10")
embedding_trainable=(1)
adam=(1)

total_steps=$((${#num_steps[@]} * ${#encoder_size[@]} * ${#decoder_size[@]} * \
${#dropout[@]} * ${#batch_size[@]} * ${#embedding_size[@]} * ${#task_embedding_size[@]} * \
${#num_shared_layers[@]} * ${#num_private_layers[@]} * ${#bidirectional[@]} * \
${#lstm[@]} * ${#embedding_path[@]} * ${#dataset_path[@]} * \
${#projection_size[@]}  * ${#task_embedding_size[@]} * ${#num_gold[@]} * ${#reg_weight[@]} * \
${#embedding_trainable[@]} * ${#adam[@]} ))

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

i=$((i % steps))
steps=$(($steps / ${#task_embedding_size[@]}))
echo ${steps}
task_embedding_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#embedding_path[@]}))
echo ${steps}
embedding_path_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#dataset_path[@]}))
echo ${steps}
dataset_path_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#projection_size[@]}))
echo ${steps}
projection_size_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#num_gold[@]}))
echo ${steps}
num_gold_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#reg_weight[@]}))
echo ${steps}
reg_weight_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#embedding_trainable[@]}))
echo ${steps}
embedding_trainable_idx=$((i / steps))

i=$((i % steps))
steps=$(($steps / ${#adam[@]}))
echo ${steps}
adam_idx=$((i / steps))

name="${dataset[dataset_path_idx]}_${num_steps[num_steps_idx]}n_\
${dropout[dropout_idx]}drop_${batch_size[batch_size_idx]}b_\
${embedding_size[embedding_size_idx]}emb_size_${lstm[lstm_idx]}cell_${mix_percent[mix_percent_idx]}mix\
${embedding[embedding_path_idx]}emb_${projection_size[projection_size_idx]}proj\
${num_gold[num_gold_idx]}num_gold_${reg_weight[reg_weight_idx]}reg_weight_\
${embedding_trainable[embedding_trainable_idx]}emb_train_${adam[adam_train_idx]}"
echo ${name}

cp -R ./data/outputs/random_lm_150_July28_conll_annealing_glove_window_proj  ./data/outputs/${directory}/${name}

LD_LIBRARY_PATH='/share/apps/mr/utils/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/mr/utils/lib6_2.17/usr/lib64/:/share/apps/gcc-5.2.0/lib64:/share/apps/gcc-5.2.0/lib:/opt/gridengine/lib/linux-x64:/opt/gridengine/lib/linux-x64:/opt/openmpi/lib:/opt/python/lib:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64:/share/apps/mr/cuda/lib:/share/apps/mr/cuda/lib64' \
  ~/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so /share/apps/mr/bin/python3 \
  ./multi-task-project/lm/run_model.py --model_type "JOINT" \
                                        --dataset_path ${dataset_path[dataset_path_idx]} \
                                        --ptb_path ${ptb_path[embedding_path_idx]} \
                                        --save_path "./data/outputs/${directory}/${name}" \
                                        --glove_path ${embedding_path[embedding_path_idx]}\
                                        --num_steps ${num_steps[num_steps_idx]} \
                                        --encoder_size ${encoder_size[encoder_size_idx]} \
                                        --pos_decoder_size ${encoder_size[encoder_size_idx]} \
                                        --chunk_decoder_size ${encoder_size[encoder_size_idx]} \
                                        --dropout ${dropout[dropout_idx]} \
                                        --batch_size ${batch_size[batch_size_idx]} \
                                        --pos_embedding_size ${task_embedding_size[task_embedding_size_idx]} \
                                        --num_shared_layers ${num_shared_layers[num_shared_idx]} \
                                        --num_private_layers ${num_private_layers[num_private_idx]} \
                                        --chunk_embedding_size ${task_embedding_size[task_embedding_size_idx]} \
                                        --lm_decoder_size  ${decoder_size[decoder_size_idx]} \
                                        --bidirectional ${bidirectional[bidirectional_idx]} \
                                        --lstm ${lstm[lstm_idx]} \
                                        --mix_percent ${mix_percent[mix_percent_idx]} \
                                        --write_to_file 1 \
                                        --embedding 1 \
                                        --max_epoch 150 \
                                        --test 0 \
                                        --projection_size ${projection_size[projection_size_idx]} \
                                        --num_gold ${num_gold[num_gold_idx]} \
                                        --reg_weight ${reg_weight[reg_weight_idx]} \
                                        --word_embedding_size ${embedding_size[embedding_size_idx]} \
                                        --embedding_trainable ${embedding_trainable[embedding_trainable_idx]} \
                                        --adam ${adam[adam_idx]}| tee "./data/outputs/${directory}/${name}.txt"
