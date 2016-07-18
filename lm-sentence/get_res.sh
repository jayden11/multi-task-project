echo 'Run Model'
#LD_LIBRARY_PATH="$HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/:$HOME/utils/libc6_2.17/usr/lib64/" \
# $HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
python3 get_predictions.py --model_type "JOINT" \
                     --dataset_path "../../data/conll" \
                     --ptb_path "../../data/ptb" \
                     --save_path "../../data/outputs/test" \
                     --glove_path '../../data/glove.6B/glove.6B.300d.txt' \
                     --num_steps 64 \
                     --encoder_size 400 \
                     --pos_decoder_size 400 \
                     --chunk_decoder_size 400 \
                     --dropout 0.7 \
                     --batch_size 128 \
                     --pos_embedding_size 100 \
                     --num_shared_layers 1 \
                     --num_private_layers 1 \
                     --chunk_embedding_size 100 \
                     --lm_decoder_size  400 \
                     --bidirectional 1 \
                     --lstm 0 \
                     --mix_percent 0.4 \
                     --write_to_file 1 \
                     --embedding 1 \
                     --max_epoch 70
