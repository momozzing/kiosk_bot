"""
python interactive.py
"""
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

model_name = "momo/gpt2-kiosk"
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
# # model.resize_token_embeddings(len(tokenizer)) 
model.cuda()

with torch.no_grad():
    while True:
        t = input("\nUser: ")
        tokens = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=50
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        sample_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=50,
            # max_new_tokens=50, 
            # top_k=50,
            # return_dict_in_generate=True
        )
        gen = sample_output[0]
        print("System: " + tokenizer.decode(gen[len(input_ids[0]):-1], skip_special_tokens=True))
