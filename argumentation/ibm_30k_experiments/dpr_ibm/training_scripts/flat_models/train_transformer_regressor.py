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
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        return tokenizer(text, padding="max_length", truncation=True)
        

    tokenized_datasets = dataset.map(tokenize_function, batched=False)
    tokenized_datasets = tokenized_datasets.map(relabel_ds, batched=False)
    dataset = tokenized_datasets[split_name]

    return dataset


def load_model(path, num_labels=3):
    class FlatModel(PreTrainedModel):
        def __init__(self, *args, **kwargs):
            config = AutoConfig.from_pretrained(path)
            super(FlatModel, self).__init__(config, *args, **kwargs)
            self.model = AutoModel.from_pretrained(path)
            self.hidden_state_dim = 768
            self.l1 = torch.nn.Linear(self.hidden_state_dim, 300)
            self.l2 = torch.nn.Linear(300, 1)
            self.gelu = torch.nn.GELU()
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,):

            out = self.model(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

            # BERT like models with poolers
            out = out[1]

            # GPT like models without pooler
            # out = out.last_hidden_state
            # out = out[:, 0, :]

            pool = self.gelu(self.l1(out))

            l1_out = self.l2(pool)
            l1_out = self.sigmoid(l1_out)
            # l2_out = self.l3(pool)
            # l3_out = self.l4(pool)
            
            return {"l1": l1_out}

    model = FlatModel()
    
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits = logits[0]
    # embed()
    # print(logits.shape, labels.shape)
    # logits1, = logits[0]
    # labels1, labels2, labels3 = labels[:, 0]
    # assert logits1.shape == labels1.shape, logits1.shape

    spr1 = spearmanr_metric.compute(predictions=logits, references=labels)["spearmanr"]
    pearson1 = pearsonr_metric.compute(predictions=logits, references=labels)["pearsonr"]

    return {
        "spearman1": spr1, 
        # "spearman2": spr2, 
        # "spearman3": spr3, 
        "pearson1": pearson1, 
        # "pearson2": pearson2, 
        # "pearson3": pearson3
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
            outputs = model(**inputs)
            logits = outputs
            # embed()
            # print(logits)
            loss_fct = torch.nn.BCELoss()

            # assert logits[0].shape == labels[0].shape, (logits[0].shape, labels[0].shape)
            # assert logits[1].shape == labels[1].shape
            # assert logits[2].shape == labels[2].shape

            # loss1 = loss_fct(logits["l1"].squeeze(-1), labels[:, 0])
            # loss2 = loss_fct(logits["l2"].squeeze(-1), labels[:, 1])
            # loss3 = loss_fct(logits["l3"].squeeze(-1), labels[:, 2])

            # loss = ((labels[0] - logits[0])**2 + (labels[1] - logits[1])**2 + (labels[2] - logits[2])**2)/3
            # print(logits["l1"].shape, labels.shape)
            loss = loss_fct(logits["l1"], labels.unsqueeze(-1))
            # loss = loss_fct(logits["l1"].view(-1, self.model.config.num_labels), labels.view(-1))
            # loss = (loss1+loss2+loss3)/3
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # with open(os.path.join(output_dir, "preds.log"), "w+") as f:
    #     f.write(json.dumps(trainer.predict(test_dataset).metrics))
    # embed()
    predictions = trainer.predict(test_dataset).predictions.squeeze(-1).tolist()
    test_df = test_dataset.to_pandas()
    test_df["predictions"] = predictions
    test_df.to_csv(f"predictions_csv.csv", index=False)

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

    # torch.cuda.set_device(1)
    # print("On device: ", torch.cuda.current_device())

    train_dataset = load_data(
        args.train_dataset_path,
        # [args.train_dataset_path, "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/review_train.csv", "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/qa_train.csv"],
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="train",
    ).shuffle(5000)

    val_dataset = load_data(
        args.val_dataset_path,
        # [args.val_dataset_path, "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/review_val.csv", "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/qa_val.csv"],
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="val",
    )

    test_dataset = load_data(
        args.test_dataset_path,
        # [args.test_dataset_path, "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/review_test.csv", "/scratch/darshan/forecast-argument/datasets/gaq_cleaned/qa_test.csv"],
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="test",
    )

    spearmanr_metric = evaluate.load("spearmanr")
    pearsonr_metric = evaluate.load("pearsonr")

    model = load_model(args.model_name, num_labels=1)

    train(train_dataset, val_dataset, model, args.output_dir)
