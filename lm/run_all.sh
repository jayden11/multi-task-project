echo 'Run Model'
#LD_LIBRARY_PATH="$HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/:$HOME/utils/libc6_2.17/usr/lib64/" \
# $HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
python3 run_model.py --model_type "JOINT" \
                     --dataset_path "../../data/conll" \
                     --ptb_path "../../data/ptb" \
                     --save_path "../../data/outputs/test" \
                     --num_steps 200 \
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
                     --bidirectional "False" \
                     --lstm "False" \
                     --write_to_file "True"
