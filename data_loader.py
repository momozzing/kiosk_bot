import json # import json module
import pandas as pd 
import numpy as np

# with statement
with open("data/SDRW2000000001.json",'r', encoding='UTF-8') as json_file:
    json_data = json.load(json_file)

# print(json_data)

# print(json_data['document'][0]['utterance'])

json_data = pd.DataFrame(json_data['document'][0]["utterance"], 
                         columns = ['id', 'form', 'original_form', 'speaker_id', 'start', 'end', 'note'])


form_json_data= json_data.drop(['id', 'start', 'end','note','original_form'], axis=1)
original_form_json_data= json_data.drop(['id', 'start', 'end','note','form'], axis=1)

print(form_json_data.shape)
print(original_form_json_data.shape)


# print(json_data)

# json_data.set_index('speaker_id', inplace = True)

# print(json_data)

# for form, original_form in zip(json_data['form'], json_data['original_form']):
#     print(form, original_form)
    # if text['speaker_id'] == text['speaker_id']: 

    #     print(text)

    #     tmp.append(json_data['form'], json_data['original_form'])

    # print(tmp)

for line in range(0, len(form_json_data)):
    # for id in json_data['speaker_id']:
    if form_json_data['speaker_id'][line] == form_json_data['speaker_id'][line+1]:
        tmp = []
        tmp.append(form_json_data['form'].values)

    elif form_json_data['speaker_id'].empty != form_json_data['speaker_id'].empty:
        break
print(tmp)

tmp = pd.DataFrame(tmp)

print(tmp)
# test = json_data['form'][0] + json_data['form'][1]
# print(test)