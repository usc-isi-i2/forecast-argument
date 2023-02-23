import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
import argparse
import json
from tqdm import tqdm
import torch

tqdm.pandas()
torch.set_grad_enabled(False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
searcher_faiss = FaissSearcher(
    '/nas/home/darshang/dpr_index',
    'facebook/dpr-question_encoder-multiset-base'
)
doc_retriever = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")

def get_similar(query, method="bert", num_items=2, sep_token="[SEP]"):

    if method == "bert":
        hits = searcher_faiss.search(query, k=num_items)
        context = ""
        for i in range(num_items):
            d = json.loads(doc_retriever.doc(hits[i].docid).raw())['contents']
            if i != num_items - 1:
                context += d + sep_token + " "
            else:
                context += d
        return context

    else:
        hits = searcher.search(query, k=num_items)
        context = ""
        for i in range(num_items):
            if i != num_items - 1:
                context += hits[i].raw + sep_token + " "
            else:
                context += hits[i].raw
        return context


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="train_ibm_30k.csv", type=str, required=False)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--col_name", type=str, required=False, default="title")

    args = parser.parse_args()

    df = read_dataset(args.dataset_path)
    df["similar"] = df[args.col_name].progress_map(get_similar,)
    df.to_csv(args.out_path, index=False)
