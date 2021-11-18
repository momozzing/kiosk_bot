import pandas as pd 
from IPython.display import display
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

session_num = 0
new_df['dialog_session'] = 0

for i in range(len(new_df)):
    if new_df["SENTENCEID"][i] == '1':
        session_num += 1
        new_df['dialog_session'][i] =+ session_num

print(new_df)
