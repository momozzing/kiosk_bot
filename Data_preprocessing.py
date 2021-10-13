import pandas as pd 
from tqdm import tqdm

train_data = pd.read_csv("data/cafe_data.txt", delimiter="\t", encoding= "utf-8")

tmp = []
for idx in tqdm(range(len(train_data))):
    sentense, speakerid, sentenseid = train_data["SENTENCE"][idx], train_data["SPEAKERID"][idx], train_data["SENTENCEID"][idx]
    tmp.append([sentense] + [speakerid] + [sentenseid])
 
new_df = pd.DataFrame(tmp, columns=['SENTENCE', 'SPEAKERID', 'SENTENCEID'])
new_df = new_df[:7180]

text_list = []
label_list = []
all_list = []

for i in range(len(new_df)):
    if new_df["SPEAKERID"][i] == 1:
        text = new_df["SENTENCE"][i]
        text_list.append(text)
    elif new_df["SPEAKERID"][i] == 0:
        label = new_df["SENTENCE"][i]
        label_list.append(label)
    
    
text_data = pd.DataFrame(text_list, columns=['text'])
label_data = pd.DataFrame(label_list, columns=['label'])

all_data = pd.concat([text_data,label_data],axis=1)

all_data.to_csv('data/cafe_clear_data', index=False, sep="\t")