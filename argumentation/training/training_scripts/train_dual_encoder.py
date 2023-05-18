import argparse
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Features, Value
import evaluate
import os
import json
from transformers import PreTrainedModel, AutoConfig, AutoModel
from typing import Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_data(
    path,
    tokenizer_name="",
    split_name="train",
    append_context="stack",
    max_length=512,
    sep_token="[SEP]",
):
    features = Features(
        {
            "cogency_mean": Value("float"),
            "effectiveness_mean": Value("float"),
            "reasonableness_mean": Value("float"),
            "text": Value("string"),
            "title": Value("string"),
            "feedback": Value("string"),
            "assumption": Value("string"),
            "counter": Value("string"),
            "similar": Value("string"),
        }
    )

    dataset = load_dataset(
        "csv",
        data_files={
            split_name: path,
        },
        delimiter=",",
        column_names=[
            "cogency_mean",
            "effectiveness_mean",
            "reasonableness_mean",
            "text",
            "title",
            "feedback",
            "assumption",
            "counter",
            "similar",
        ],
        skiprows=1,
        features=features,
        keep_in_memory=True,
    )

    # Load tokenizer and tokenize
    if "xlnet" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=2048)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if "gpt" in tokenizer_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.sep_token = "\n"
    if "transfo-xl" in tokenizer_name:
        tokenizer.pad_token = "[PAD]"

    def relabel_ds(examples):
        examples["label"] = [
            float(examples["cogency_mean"]) / 5,
            float(examples["effectiveness_mean"]) / 5,
            float(examples["reasonableness_mean"]) / 5,
        ]
        return examples

    def stack_dict_elements(dicts):
        stacked_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in stacked_dict:
                    stacked_dict[key].append(value)
                else:
                    stacked_dict[key] = [value]
        return stacked_dict

    def tokenize_stack(examples):
        text = (
            f"""Topic: {examples["title"]} {sep_token} Argument: {examples["text"]}"""
        )
        if torch.rand(1).item() < 0.5:
            context = f"""Similar stance: {examples["similar"]} {sep_token} Feedback: {examples["feedback"]} {sep_token} Assumptions: {examples["assumption"]} {sep_token} Counter argument: {examples["counter"]}"""

            # Uncomment for individual augmentations
            # context = f"""Similar stance: {examples["similar"]}"""
            # context = (f'''Feedback: {examples["feedback"]}''')
            # context = (f'''Assumptions: {examples["assumption"]}''')
            # context = (f'''Counter argument: {examples["counter"]}''')
        else:
            context = f"""Similar stance: {"None"} {sep_token} Feedback: {examples["feedback"]} {sep_token} Assumptions: {examples["assumption"]} {sep_token} Counter argument: {examples["counter"]}"""

            # Uncomment for individual augmentations
            # context = f"""Similar stance: {"None"}"""
            # context = (f'''Feedback: {examples["feedback"]}''')
            # context = (f'''Assumptions: {examples["assumption"]}''')
            # context = (f'''Counter argument: {examples["counter"]}''')

        sent = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_length
        )
        context = tokenizer(
            context, padding="max_length", truncation=True, max_length=max_length
        )
        return stack_dict_elements([sent, context])
        # return {"argument": sent, "context": context}

    def tokenize_append(examples):
        if torch.rand(1).item() < 0.5:
            context = f"""Similar stance: {examples["similar"]} {sep_token} Feedback: {examples["feedback"]} {sep_token} Assumptions: {examples["assumption"]} {sep_token} Counter argument: {examples["counter"]}"""

            # Uncomment for individual augmentations
            # context = (f'''Similar stance: {examples["similar"]}''')
            # context = (f'''Feedback: {examples["feedback"]}''')
            # context = (f'''Assumptions: {examples["assumption"]}''')
            # context = (f'''Counter argument: {examples["counter"]}''')
        else:
            context = f"""Similar stance: {"None"} {sep_token} Feedback: {examples["feedback"]} {sep_token} Assumptions: {examples["assumption"]} {sep_token} Counter argument: {examples["counter"]}"""

            # Uncomment for individual augmentations
            # context = (f'''Similar stance: {"None"}''')
            # context = (f'''Feedback: {examples["feedback"]}''')
            # context = (f'''Assumptions: {examples["assumption"]}''')
            # context = (f'''Counter argument: {examples["counter"]}''')
        text = (
            f"""Topic: {examples["title"]} {sep_token} Argument: {examples["text"]}"""
        )
        return tokenizer(
            text + f" {sep_token} " + context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def tokenize_text(examples):
        text = (
            f"""Topic: {examples["title"]} {sep_token} Argument: {examples["text"]}"""
        )

        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            # padding=True,
        )

    if append_context == "append":
        tokenized_datasets = dataset.map(tokenize_append, batched=False)
    elif append_context == "stack":
        tokenized_datasets = dataset.map(tokenize_stack, batched=False)
    else:
        tokenized_datasets = dataset.map(tokenize_text, batched=False)

    tokenized_datasets = tokenized_datasets.map(relabel_ds, batched=False)
    dataset = tokenized_datasets[split_name]
    return dataset


def load_ibm(
    path,
    tokenizer_name="",
    split_name="train",
    append_context="stack",
    max_length=512,
    sep_token="[SEP]",
):
    features = Features(
        {
            "argument": Value("string"),
            "topic": Value("string"),
            "set": Value("string"),
            "WA": Value("float"),
            "MACE-P": Value("float"),
            "stance_WA": Value("float"),
            "stance_WA_conf": Value("float"),
            "Feedback": Value("string"),
            "Assumption": Value("string"),
            "Counter": Value("string"),
        }
    )

    dataset = load_dataset(
        "csv",
        data_files={
            split_name: path,
        },
        delimiter=",",
        column_names=[
            "argument",
            "topic",
            "set",
            "WA",
            "MACE-P",
            "stance_WA",
            "stance_WA_conf",
            "Feedback",
            "Assumption",
            "Counter",
        ],
        skiprows=1,
        features=features,
        keep_in_memory=True,
    )

    # Load tokenizer and tokenize
    if "xlnet" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=2048)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.max_length = 4096

    if "gpt" in tokenizer_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.sep_token = "\n"
    if "transfo-xl" in tokenizer_name:
        tokenizer.pad_token = "[PAD]"

    def relabel_ds(examples):
        examples["label"] = None
        return examples

    def stack_dict_elements(dicts):
        stacked_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in stacked_dict:
                    stacked_dict[key].append(value)
                else:
                    stacked_dict[key] = [value]
        return stacked_dict

    def tokenize_stack(examples):
        text = f"""Topic: {examples["topic"]} {sep_token} Argument: {examples["argument"]}"""

        context = f"""Similar stance: {"None"} {sep_token} Feedback: {examples["Feedback"]} {sep_token} Assumptions: {examples["Assumption"]} {sep_token} Counter argument: {examples["Counter"]}"""

        # Uncomment for individual augmentations
        # context = f"""Similar stance: {"None"}"""
        # context = (f'''Feedback: {examples["Feedback"]}''')
        # context = (f'''Assumptions: {examples["Assumption"]}''')
        # context = (f'''Counter argument: {examples["Counter"]}''')

        sent = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_length
        )
        context = tokenizer(
            context, padding="max_length", truncation=True, max_length=max_length
        )
        return stack_dict_elements([sent, context])

    def tokenize_append(examples):
        context = f"""Similar stance: {"None"} {sep_token} Feedback: {examples["Feedback"]} {sep_token} Assumptions: {examples["Assumption"]} {sep_token} Counter argument: {examples["Counter"]}"""

        # Uncomment for individual augmentations
        # context = (f'''Similar stance: {"None"}''')
        # context = (f'''Feedback: {examples["feedback"]}''')
        # context = (f'''Assumptions: {examples["assumption"]}''')
        # context = (f'''Counter argument: {examples["counter"]}''')

        text = f"""Topic: {examples["topic"]} {sep_token} Argument: {examples["argument"]}"""
        return tokenizer(
            text + f" {sep_token} " + context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def tokenize_text(examples):
        text = f"""Topic: {examples["topic"]} {sep_token} Argument: {examples["argument"]}"""

        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    if append_context == "append":
        tokenized_datasets = dataset.map(tokenize_append, batched=False)
    elif append_context == "stack":
        tokenized_datasets = dataset.map(tokenize_stack, batched=False)
    else:
        tokenized_datasets = dataset.map(tokenize_text, batched=False)

    tokenized_datasets = tokenized_datasets.map(relabel_ds, batched=False)
    dataset = tokenized_datasets[split_name]
    return dataset


def load_model(
    path,
    num_labels=3,
    tokenizer_name="bert-base-uncased",
    # pool_mode="mean",
    pool_mode="mean",
    dropout=True,
    dropout_rate=0.3,
    use_sigmoid=False,
    use_dual_encoder=True,
    freeze_encoders=False,
    use_projection=True,
    use_att=True,
):
    class FlatModel(PreTrainedModel):
        def __init__(self, tokenizer_name, *args, **kwargs):
            config = AutoConfig.from_pretrained(path)
            super(FlatModel, self).__init__(config, *args, **kwargs)

            self.hidden_state_dim = config.hidden_size
            self.use_dual_encoder = use_dual_encoder
            self.tokenizer_name = tokenizer_name
            self.use_sigmoid = use_sigmoid
            self.use_projection = use_projection
            self.use_att = True

            self.encoder = AutoModel.from_pretrained(path)
            if self.use_dual_encoder:
                self.encoder2 = AutoModel.from_pretrained(path)

            if freeze_encoders:
                for param in self.encoder.parameters():
                    param.requires_grad = False

                for param in self.encoder2.parameters():
                    param.requires_grad = False

            if self.use_projection:
                self.l1 = torch.nn.Linear(self.hidden_state_dim, 300)
                head_input_shape = 300
            else:
                head_input_shape = self.hidden_state_dim

            self.l2 = torch.nn.Linear(head_input_shape, 1)
            self.l3 = torch.nn.Linear(head_input_shape, 1)
            self.l4 = torch.nn.Linear(head_input_shape, 1)

            self.l2.weight.data.fill_(0.0)
            self.l2.bias.data.fill_(0.0)

            self.l3.weight.data.fill_(0.0)
            self.l3.bias.data.fill_(0.0)

            self.l4.weight.data.fill_(0.0)
            self.l4.bias.data.fill_(0.0)

            self.gelu = torch.nn.GELU()
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout(dropout_rate)

            self.att = torch.nn.MultiheadAttention(
                self.hidden_state_dim,
                num_heads=8,
                batch_first=True,
            )

        def mean_pooling(self, model_output, attention_mask):
            # token_embeddings = model_output[
            #     0
            # ]  # First element of model_output contains all token embeddings
            token_embeddings = model_output
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def mean_pooling_dual_encoder(
            self, model_output, model_output1, attention_mask, attention_mask1
        ):
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            )
            sum = torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

            input_mask_expanded1 = (
                attention_mask1.unsqueeze(-1).expand(model_output1.size()).float()
            )
            sum1 = torch.sum(model_output1 * input_mask_expanded1, 1) / torch.clamp(
                input_mask_expanded1.sum(1), min=1e-9
            )

            return sum + sum1

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):

            if self.use_dual_encoder:
                input_ids, input_ids1 = input_ids[:, 0, :], input_ids[:, 1, :]
                attention_mask, attention_mask1 = (
                    attention_mask[:, 0, :],
                    attention_mask[:, 1, :],
                )
                if token_type_ids is None:
                    token_type_ids, token_type_ids1 = None, None
                else:
                    token_type_ids, token_type_ids1 = (
                        token_type_ids[:, 0, :],
                        token_type_ids[:, 1, :],
                    )

                out1 = self.encoder2(
                    input_ids1,
                    attention_mask=attention_mask1,
                    return_dict=return_dict,
                )

            out = self.encoder(
                input_ids,
                return_dict=return_dict,
            )

            if pool_mode == "mean":
                if self.use_dual_encoder:
                    if self.use_att:
                        att_out, _ = self.att(
                            out.last_hidden_state,
                            out1.last_hidden_state,
                            out1.last_hidden_state,
                        )
                        out = self.mean_pooling_dual_encoder(
                            out.last_hidden_state + out1.last_hidden_state + att_out,
                            out.last_hidden_state + out1.last_hidden_state + att_out,
                            attention_mask,
                            attention_mask1,
                        )
                    else:
                        out = self.mean_pooling_dual_encoder(
                            out.last_hidden_state,
                            out1.last_hidden_state,
                            attention_mask,
                            attention_mask1,
                        )
                else:
                    out = self.mean_pooling(out.last_hidden_state, attention_mask)
                pool = out

                if self.use_projection:
                    pool = self.gelu(self.l1(pool))
            else:
                out = out.last_hidden_state
                out = out[:, 0, :]

                if self.use_dual_encoder:
                    out1 = out1.last_hidden_state
                    out1 = out1[:, 0, :]

                    if self.use_att:
                        attn_output, _ = self.att(out, out1, out1)
                        pool = self.gelu(self.l1(attn_output))
                    else:
                        pool = self.gelu(self.l1(out + out1))
                else:
                    pool = self.gelu(self.l1(out))

            if dropout:
                pool = self.dropout(pool)

            l1_out = self.l2(pool)
            l2_out = self.l3(pool)
            l3_out = self.l4(pool)

            if self.use_sigmoid:
                l1_out = self.sigmoid(l1_out)
                l2_out = self.sigmoid(l2_out)
                l3_out = self.sigmoid(l3_out)

            return {"l1": l1_out, "l2": l2_out, "l3": l3_out}

    model = FlatModel(tokenizer_name)

    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits1, logits2, logits3 = logits[0], logits[1], logits[2]
    labels1, labels2, labels3 = labels[:, 0], labels[:, 1], labels[:, 2]

    spr1 = spearmanr_metric.compute(predictions=logits1, references=labels1)[
        "spearmanr"
    ]
    spr2 = spearmanr_metric.compute(predictions=logits2, references=labels2)[
        "spearmanr"
    ]
    spr3 = spearmanr_metric.compute(predictions=logits3, references=labels3)[
        "spearmanr"
    ]

    pearson1 = pearsonr_metric.compute(predictions=logits1, references=labels1)[
        "pearsonr"
    ]
    pearson2 = pearsonr_metric.compute(predictions=logits2, references=labels2)[
        "pearsonr"
    ]
    pearson3 = pearsonr_metric.compute(predictions=logits3, references=labels3)[
        "pearsonr"
    ]

    return {
        "spearman1": spr1,
        "spearman2": spr2,
        "spearman3": spr3,
        "pearson1": pearson1,
        "pearson2": pearson2,
        "pearson3": pearson3,
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
        metric_for_best_model="eval_pearson1",
        greater_is_better=True,
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs
            loss_fct = torch.nn.MSELoss()

            loss1 = loss_fct(
                torch.clamp(logits["l1"], 0, 1), labels[:, 0].unsqueeze(-1)
            )
            loss2 = loss_fct(
                torch.clamp(logits["l2"], 0, 1), labels[:, 1].unsqueeze(-1)
            )
            loss3 = loss_fct(
                torch.clamp(logits["l3"], 0, 1), labels[:, 2].unsqueeze(-1)
            )

            loss = loss1 + loss2 + loss3
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    with open(os.path.join(output_dir, "preds_all.log"), "w+") as f:
        f.write(json.dumps(trainer.predict(test_dataset).metrics))

    ibm_preds = trainer.predict(test_ibm_dataset)

    a1, a2, a3 = (
        ibm_preds.predictions[0],
        ibm_preds.predictions[1],
        ibm_preds.predictions[2],
    )
    df = pd.read_csv("datasets/gpt_gaq/augmented_arg_quality_rank_30k.csv")
    wa = df["WA"].to_numpy()

    print(pearsonr(wa, (a1 + a2 + a3) / 3))
    print(spearmanr(wa, (a1 + a2 + a3) / 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_path",
        default="train_dataset.csv",
        type=str,
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--val_dataset_path",
        default="val_dataset.csv",
        type=str,
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--test_dataset_path",
        default="test_dataset.csv",
        type=str,
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--test_ibm_dataset_path",
        default="test_ibm_dataset.csv",
        type=str,
        required=True,
    )
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default="output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", default=16, type=int, required=True)
    parser.add_argument("--eval_batch_size", default=16, type=int, required=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=True)
    parser.add_argument("--epochs", default=5, type=int, required=True)

    args = parser.parse_args()

    train_dataset = load_data(
        args.train_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="train",
        max_length=None,
        append_context="stack",
    ).shuffle(5000)

    val_dataset = load_data(
        args.val_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="val",
        max_length=None,
        append_context="stack",
    )

    test_dataset = load_data(
        args.test_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="test",
        max_length=None,
        append_context="stack",
    )

    test_ibm_dataset = load_ibm(
        args.test_ibm_dataset_path,
        args.tokenizer if args.tokenizer else args.model_name,
        split_name="test",
        max_length=None,
        append_context="stack",
    )

    spearmanr_metric = evaluate.load("spearmanr")
    pearsonr_metric = evaluate.load("pearsonr")

    model = load_model(
        args.model_name,
        tokenizer_name=args.tokenizer if args.tokenizer else args.model_name,
        dropout=True,
        dropout_rate=0.5,
        use_sigmoid=False,
        use_dual_encoder=True,
        freeze_encoders=False,
        use_att=True,
    )

    train(train_dataset, val_dataset, model, args.output_dir)
