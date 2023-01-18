pwd

source activate virtual-env/bin/source

python3 train_transformer_regressor.py --train_dataset_path="datasets/train_ibm_30k.csv" \
                                        --val_dataset_path="datasets/val_ibm_30k.csv" \
                                        --test_dataset_path="datasets/test_ibm_30k.csv" \
                                        --model_name="roberta-base" \
                                        --output_dir="output_dir" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5

python3 train_transformer_regressor.py --train_dataset_path="datasets/train_ibm_30k.csv" \
                                        --val_dataset_path="datasets/val_ibm_30k.csv" \
                                        --test_dataset_path="datasets/test_ibm_30k.csv" \
                                        --model_name="bert-base-uncased" \
                                        --output_dir="output_dir" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5

python3 train_transformer_regressor.py --train_dataset_path="datasets/train_ibm_30k.csv" \
                                        --val_dataset_path="datasets/val_ibm_30k.csv" \
                                        --test_dataset_path="datasets/test_ibm_30k.csv" \
                                        --model_name="chkla/roberta-argument" \
                                        --output_dir="output_dir" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5

deactivate