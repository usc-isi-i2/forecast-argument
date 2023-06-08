# Argument Quality Analysis

## Install requirements
To install all necessary requirements, run the following script:

```bash
pip install -r requirements.txt
```

## Reproducing results
1. Running the DPR training

```bash
python3 train_dpr.py --train_dataset_path="path1/" "path2/" "path3/" \
                                        --val_dataset_path="path1/" "path2/" "path3/" \
                                        --test_dataset_path="path1/" "path2/" "path3/" \
                                        --test_ibm_dataset_path="path/" \
                                        --model_name="bert-base-uncased" \
                                        --output_dir="output_dir/" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5
```
2. Running the custom augmentation training

```bash
python3 train_dual_encoder.py --train_dataset_path="path1/" "path2/" "path3/" \
                                        --val_dataset_path="path1/" "path2/" "path3/" \
                                        --test_dataset_path="path1/" "path2/" "path3/" \
                                        --test_ibm_dataset_path="path/" \
                                        --model_name="bert-base-uncased" \
                                        --output_dir="output_dir/" \
                                        --train_batch_size=16 \
                                        --eval_batch_size=16 \
                                        --learning_rate=5e-5 \
                                        --epochs=5
```

3. Augmenting the dataset
The augmenting script, `augment.py` requires three input arguments: the input csv file, the output directory and an optional boolean argument which determines if a similar quality argument is to be sampled or not. Sampling a similar quality argument requires labels so this must not be used for testing splits.
```bash
python3 augment.py --input_file="path/to/csv" \
                    --output_dir="path/to/output_dir" \
                    --add_similar
```
