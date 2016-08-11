echo 'Run Model'
#LD_LIBRARY_PATH="$HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/:$HOME/utils/libc6_2.17/usr/lib64/" \
# $HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
python3 run_model.py --model_type "JOINT" \
                     --dataset_path "../../data/conll_toy" \
                     --ptb_path "../../data/conll_toy" \
                     --save_path "../../data/outputs/test" \
                     --glove_path '../../data/glove.6B/glove.6B.50d.txt' \
                     --num_steps 10 \
                     --encoder_size 200 \
                     --pos_decoder_size 200 \
                     --chunk_decoder_size 200 \
                     --dropout 0.5 \
                     --batch_size 64 \
                     --pos_embedding_size 400 \
                     --num_shared_layers 1 \
                     --num_private_layers 1 \
                     --chunk_embedding_size 400 \
                     --lm_decoder_size  200 \
                     --bidirectional 1 \
                     --lstm 1 \
                     --mix_percent 1 \
                     --write_to_file 0 \
                     --embedding 1 \
                     --max_epoch 1 \
                     --test 0 \
                     --num_gold 1 \
                     --reg_weight 1e-09 \
                     --word_embedding_size 50 \
                     --embedding_trainable 1 \
                     --projection_size 25 \
                     --adam 1
