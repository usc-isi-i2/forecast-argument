eval "$(conda shell.bash hook)"
conda activate virtual-env

CUDA_VISIBLE_DEVICES=0 python pyserini_augment.py --dataset_path="/scratch/darshan/forecast-argument/datasets/gaq_cleaned/qa_test.csv" --out_path="/scratch/darshan/forecast-argument/datasets/gaq_dpr/qa_test.csv"
echo "Done with test 1"



conda deactivate
