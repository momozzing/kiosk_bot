import pandas as pd 
from tqdm import tqdm

train_data = pd.read_csv("data/cafe_data.txt", delimiter="\t", encoding= "utf-8")

tmp = []
for idx in tqdm(range(len(train_data))):
    sentense, speakerid, sentenseid = train_data["SENTENCE"][idx], train_data["SPEAKERID"][idx], train_data["SENTENCEID"][idx]
    tmp.append([sentense] + [speakerid] + [sentenseid])
 
new_df = pd.DataFrame(tmp, columns=['SENTENCE', 'SPEAKERID', 'SENTENCEID'])

# print(new_df)

# print(new_df.isnull().sum())


text = []
label = []

session_df = new_df.set_index('session.ID')

for i in range(len(new_df)):
    ithSessionDF=new_df.loc[i][['SENTENCE','SPEAKERID','SENTENCEID']]

    # if not isinstance(ithSessionDF, pd.DataFrame):
    #     continue
    # ithSessionDF= ithSessionDF.reset_index()

    current_data = ""
    for idx in range(len(ithSessionDF)):
        line = ithSessionDF.loc[idx]
        print(line)
        current_data = current_data + " " + line["SENTENCE"]

        print(current_data)

        # if idx == 1: ## session.ID가 홀수일때 
        #     text.append(current_data)
        #     label.append(line["SENTENCE"])

# print(new_df['SPEAKERID']) 

# for i in range(len(new_df)):
#     if new_df['SPEAKERID'][i] % 2 == 1:
#         current_data = new_df['SENTENCE'][i]
#     else:
#         label_data = new_df['SENTENCE'][i]

#     print(label_data)



# for i in range(1, len(train_data)):
#     ithSessionDF=new_df.loc[i][['SENTENCE','SPEAKERID','SENTENCEID']]

#     # if not isinstance(ithSessionDF, pd.DataFrame):
#     #     continue
#     # ithSessionDF= ithSessionDF.reset_index()

#     # print(ithSessionDF)

#     current_data = ""
#     for idx in range(len(ithSessionDF)):
#         line = ithSessionDF.loc[idx]
#         current_data = current_data + " " + line["SENTENCE"]

#         print(current_data)


# text_data = pd.DataFrame(text, columns=['text'])
# label_data = pd.DataFrame(label, columns=['label'])

# all_data = pd.concat([text_data,label_data],axis=1)

# print(all_data)