import pandas as pd
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Features, Value
import evaluate
import os
import json
import torch
from IPython import embed
from typing import Optional
from transformers import PreTrainedModel, AutoConfig


def load_data(path, tokenizer_name="", split_name="train"):
    features = Features(
        {
            "argument": Value("string"),
            "topic": Value("string"),
            "WA": Value("float"),
            "similar": Value("string")
        }
    )

    dataset = load_dataset(
        "csv",
        data_files={
            split_name: path,
        },
        delimiter=",",
        column_names=["argument", "topic", "WA", "similar"],
        skiprows=1,
        features=features,
        keep_in_memory=True,
    )
    print("-"*100)
    print(dataset)

    # Load tokenizer and tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if "gpt" in tokenizer_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    def relabel_ds(examples):
        # print(examples)
        examples["label"] = float(examples["WA"])
        return examples

    def tokenize_function(examples):
        text = (
            "Topic:"
            + examples["topic"] if examples["topic"] is not None else ""
            + " "
            + tokenizer.sep_token if tokenizer.sep_token else "\n"
            + " Argument:"
            + examples["argument"] if examples["argument"] is not None else ""
            + tokenizer.sep_token if tokenizer.sep_token else "\n"
            + "Context:"
            + examples["similar"]
        )
        return tokenizer(text, padding="max_length", truncation=True)
        

    tokenized_datasets = dataset.map(tokenize_function, batched=False)
    tokenized_datasets = tokenized_datasets.map(relabel_ds, batched=False)
    dataset = tokenized_datasets[split_name]

    return dataset


def load_model(path, num_labels=1):
    model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            problem_type="regression",
        )    
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    spr1 = spearmanr_metric.compute(predictions=logits, references=labels)["spearmanr"]
    pearson1 = pearsonr_metric.compute(predictions=logits, references=labels)["pearsonr"]

    return {
        "spearman1": spr1, 
        "pearson1": pearson1
        }


def train(train_dataset, val_dataset, model, output_dir):
    # Define training arguments for trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            logits = model(**inputs)
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(logits["l1"], labels.unsqueeze(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions = trainer.predict(test_dataset).predictions.squeeze(-1).tolist()
    test_df = test_dataset.to_pandas()
    test_df["predictions"] = predictions
    test_df.to_csv(f"{output_dir}/predictions_csv.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_arg_quality_rank_30k.zip
    parser.add_argument(
        "--train_dataset_path", default="train.csv", type=str, required=True
    )
    parser.add_argument(
        "--val_dataset_path", default="val.csv", type=str, required=True
    )
    parser.add_argument(
        "--test_dataset_path", default="test.csv", type=str, required=True
    )
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default="output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", default=16, type=int, required=True)
    parser.add_argument("--eval_batch_size", default=16, type=int, required=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)
    parser.add_argument("--device", default=0, type=int, required=False)

    args = parser.parse_args()

    train_dataset = load_data(
        args.train_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="train",
    ).shuffle(5000)

    val_dataset = load_data(
        args.val_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="val",
    )

    test_dataset = load_data(
        args.test_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="test",
    )

    spearmanr_metric = evaluate.load("spearmanr")
    pearsonr_metric = evaluate.load("pearsonr")

    model = load_model(args.model_name, num_labels=1)

    train(train_dataset, val_dataset, model, args.output_dir)
