'''
deepspeed --num_gpus=1 GPT-2_fine_tune.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<seq>"
    }
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelWithLMHead.from_pretrained(
    model_name
).cuda()

model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--bos_token", default=tokenizer.bos_token, type=str)
parser.add_argument("--eos_token", default=tokenizer.eos_token, type=str)
args = parser.parse_args()

wandb.init(project="mobot", name=f"mobot-{model_name}")
train_data = pd.read_csv("data/cafe_clear_data_test.tsv", delimiter="\t")
train_data = train_data[:3000]
train_text, train_labels = (
    train_data["text"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t + str(args.bos_token) + l + str(args.eos_token), "label": l}
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/cafe_clear_data_test.tsv", delimiter="\t")
eval_data = eval_data[3000:]

eval_text, eval_labels = (
    eval_data["text"].values,
    eval_data["label"].values,
)

dataset = [
    {"data": t + str(args.bos_token) + l + str(args.eos_token), "label": l }
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = DeepSpeedCPUAdam(
    lr=3e-5, weight_decay=3e-7, model_params=model.parameters()
)

engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, optimizer=optimizer
)
epochs = 0
for epoch in range(args.epoch):
    epochs += 1
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"]
        text_tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=100,
            truncation=True,
            padding=True,
        )
        label_tokens = tokenizer(
            label,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )

        input_ids = text_tokens.input_ids.cuda()
        # print(input_ids.shape)
        attention_mask = text_tokens.attention_mask.cuda()
        label_input_ids = label_tokens.input_ids.cuda()
        # print(label_input_ids.shape)
        # label_attention_mask = label_tokens.attention_mask.cuda()

        # input_ids = input_ids.cuda()
        # attention_mask = input_ids.cuda()

        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # output = engine.forward(
        #     input_ids=input_ids + label_input_ids,
        #     attention_mask=attention_mask + label_attention_mask,
        #     labels=label_input_ids,
        # )


        loss = output.loss
        wandb.log({"loss": loss})
        engine.backward(loss)
        optimizer.step()
        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1
        # correct = 0
        # correct += sum(classification_results.detach().cpu()==label_input_ids.detach().cpu())    


        wandb.log({"acc": acc / len(classification_results)})

    model.eval()
    for eval in tqdm(eval_loader):
        eval_text, eval_label = eval["data"], eval["label"]
        eval_text_tokens = tokenizer(
            eval_text,
            return_tensors="pt",
            max_length=100,
            truncation=True,
            padding=True,
        )
        eval_label_tokens = tokenizer(
            eval_label,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )

        input_ids = eval_text_tokens.input_ids.cuda()
        # print(input_ids.shape)
        attention_mask = eval_text_tokens.attention_mask.cuda()
        label_input_ids = eval_label_tokens.input_ids.cuda()
        # print(label_input_ids.shape)
        # label_attention_mask = label_tokens.attention_mask.cuda()

        eval_out = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        wandb.log({"eval_loss": eval_out.loss})
        classification_results = eval_out.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, eval_label):
            if res == lab:
                acc += 1
        # correct = 0
        # correct += sum(classification_results.detach().cpu()==label_input_ids.detach().cpu())    


        wandb.log({"eval_acc": acc / len(classification_results)})
        wandb.log({"epoch": epochs})

    torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-test.pt")
