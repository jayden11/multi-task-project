echo 'Run Model'
python3 run_reimplemented_model.py --model_type "JOINT" \
                    --dataset_path "../../data/genia" \
                    --ptb_path "../../data/genia_ptb" \
                    --save_path "../../data/outputs/25-genia" \
                    --glove_path '../../data/senna/senna.txt' \
                    --num_steps 32 \
                    --layer_size 200 \
                    --dropout 0.5 \
                    --batch_size 32 \
                    --connection_embedding_size 20 \
                    --mix_percent 0.7 \
	                   --word_embedding_size 50  \
                    --max_epoch 100 \
                    --fraction_of_training_data 0.25

python3 run_reimplemented_model.py --model_type "JOINT" \
                    --dataset_path "../../data/genia" \
                    --ptb_path "../../data/genia_ptb" \
                    --save_path "../../data/outputs/50-genia" \
                    --glove_path '../../data/senna/senna.txt' \
                    --num_steps 32 \
                    --layer_size 200 \
                    --dropout 0.5 \
                    --batch_size 32 \
                    --connection_embedding_size 20 \
                    --mix_percent 0.7 \
	                   --word_embedding_size 50  \
                    --max_epoch 100 \
                    --fraction_of_training_data 0.50

python3 run_reimplemented_model.py --model_type "JOINT" \
                    --dataset_path "../../data/genia" \
                    --ptb_path "../../data/genia_ptb" \
                    --save_path "../../data/outputs/75-genia" \
                    --glove_path '../../data/senna/senna.txt' \
                    --num_steps 32 \
                    --layer_size 200 \
                    --dropout 0.5 \
                    --batch_size 32 \
                    --connection_embedding_size 20 \
                    --mix_percent 0.7 \
	                   --word_embedding_size 50  \
                    --max_epoch 100 \
                    --fraction_of_training_data 0.75
