echo 'Run Model'
#LD_LIBRARY_PATH="$HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/:$HOME/utils/libc6_2.17/usr/lib64/" \
# $HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
python3 run_model.py --model_type "JOINT" \
                     --dataset_path "../../data/conll_toy" \
                     --ptb_path "../../data/conll_toy" \
                     --save_path "../../data/outputs/test" \
                     --num_steps 4 \
                     --encoder_size 200 \
                     --pos_decoder_size 200 \
                     --chunk_decoder_size 200 \
                     --dropout 0.5 \
                     --batch_size 5 \
                     --pos_embedding_size 400 \
                     --num_shared_layers 1 \
                     --num_private_layers 1 \
                     --chunk_embedding_size 400 \
                     --lm_decoder_size  200 \
                     --bidirectional 0 \
                     --lstm 1 \
                     --mix_percent 0.5 \
                     --write_to_file 1 \
                     --embedding 1