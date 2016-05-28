echo 'Run Model'
#LD_LIBRARY_PATH="$HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/:$HOME/utils/libc6_2.17/usr/lib64/" \
# $HOME/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
python3 run_model.py --model_type "JOINT" --dataset_path "../../data/conll" --ptb_path "../../data/ptb" --save_path "../../data/outputs/test"
