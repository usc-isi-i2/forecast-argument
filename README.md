# Contextualizing Argument Quality Assessment with Relevant Knowledge
This repository contains the data and additional materials for the paper **"Contextualizing Argument Quality Assessment with Relevant Knowledge"** accepted at the NAACL 2024 conference.

## Overview
In this paper, we present a novel framework, SPARK, for assessing the quality of arguments by contextualizing them with relevant external knowledge. SPARK automatically produces four types of knowledge: feedback, assumptions, counter-arguments, and similar quality arguments to help improve the performance of existing argument quality ranking models. SPARK beats all baselines both in-domain on the GAQCorpus and out-of-domain on IBM-Rank-30K.

## Install requirements
To install all requirements, run the following script:

```bash
pip install -r requirements.txt
```

## Reproducing results

1. Augmenting the dataset
The augmenting script, `augment.py` requires three input arguments: the input csv file, the output directory and an optional boolean argument which determines if a similar quality argument is to be sampled. Sampling a similar quality argument requires labels so this must not be used for testing splits.
```bash
python3 augment.py --input_file="path/to/csv" \
                    --output_dir="path/to/output_dir" \
                    --add_similar
```

For the DPR augmentation, there are three arguments that the `pyserini_augment` script takes: dataset_path (path to input csv), out_path (path to output csv) and col_name (Optional text column name. Default is "title")
```bash
python3 pyserini_augment.py --dataset_path="path/to/csv" \
                    --out_path="path/to/output_csv" \
                    --col_name="title"
```
Similarly, the `llama_augment.py` and `flan_t5_augment.py` scripts can be run by passing the input_file and output_dir to create those augmented datasets

2. Running the DPR training

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
3. Running the custom augmentation training

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

The test set evaluation results will be displayed after the training is complete.

## Citation
If you use the data or any materials from this repository, please cite our paper:
```bibtex
@misc{deshpande2023contextualizing,
      title={Contextualizing Argument Quality Assessment with Relevant Knowledge},
      author={Darshan Deshpande and Zhivar Sourati and Filip Ilievski and Fred Morstatter},
      year={2023},
      eprint={2305.12280},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
For any questions or issues, please contact Darshan Deshpande at darshang@isi.edu.
