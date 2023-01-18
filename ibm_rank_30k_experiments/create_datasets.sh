pwd 

source activate virtual-env/bin/activate

python create_datasets.py --full_dataset="datasets/arg_quality_rank_30k.csv" --output_dir="datasets/"

deactivate