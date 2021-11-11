"""
python interactive.py
"""

from argparse import ArgumentParser
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer

model_name = "skt/kogpt2-base-v2"
ckpt_name = "model_save/skt-kogpt2-base-v2.pt"
model = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<seq>"
    }
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
tokenizer.add_special_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer)) 



parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

while True:
    t = input("\nText: ")
    # q = input("Question: ")
    tokens = tokenizer(
        t,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    # classification_results = output.logits.argmax(-1)
    sample_output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=50, 
        top_k=50
    )
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))



    # gen = tokenizer.convert_ids_to_tokens(
    #     torch.argmax(
    #         output,
    #         dim=-1).detach().cpu().squeeze().numpy().tolist())[-1]

    # gen = tokenizer.convert_ids_to_tokens((classification_results[0]))

    # print(f"Result: {gen}")
