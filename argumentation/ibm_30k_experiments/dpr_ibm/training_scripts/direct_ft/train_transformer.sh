pwd

eval "$(conda shell.bash hook)"
conda activate virtual-env

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train_transformer_regressor.py --train_dataset_path="/scratch/darshan/forecast-argument/argumentation/datasets/ibm_30k/ibm_30k_dpr/train_set.csv" \
                                        --val_dataset_path="/scratch/darshan/forecast-argument/argumentation/datasets/ibm_30k/ibm_30k_dpr/val_set.csv" \
                                        --test_dataset_path="/scratch/darshan/forecast-argument/argumentation/datasets/ibm_30k/ibm_30k_dpr/test_set.csv" \
                                        --model_name="roberta-base" \
                                        --output_dir="/scratch/darshan/forecast-argument/argumentation/ibm_30k_experiments/dpr_ibm/checkpoints/direct_ft_checkpoints/roberta_base_model" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5

# python3 train_transformer_regressor.py --train_dataset_path="datasets/train_ibm_30k.csv" \
#                                         --val_dataset_path="datasets/val_ibm_30k.csv" \
#                                         --test_dataset_path="datasets/test_ibm_30k.csv" \
#                                         --model_name="bert-base-uncased" \
#                                         --output_dir="output_dir" \
#                                         --train_batch_size=16 \
#                                         --eval_batch_size=16 \
#                                         --learning_rate=5e-5 \
#                                         --epochs=5

# python3 train_transformer_regressor.py --train_dataset_path="datasets/train_ibm_30k.csv" \
#                                         --val_dataset_path="datasets/val_ibm_30k.csv" \
#                                         --test_dataset_path="datasets/test_ibm_30k.csv" \
#                                         --model_name="chkla/roberta-argument" \
#                                         --output_dir="output_dir" \
#                                         --train_batch_size=16 \
#                                         --eval_batch_size=16 \
#                                         --learning_rate=5e-5 \
#                                         --epochs=5

conda deactivate