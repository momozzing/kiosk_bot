import json # import json module
import pandas as pd 
import numpy as np

# with statement
# with open("korean_dialogue_history_summary/Training/Train/education/education.json",'r', encoding='UTF-8') as json_file:
#     json_data = json.load(json_file)

json_data = open('korean_dialogue_history_summary/Training/Train/education/education.json').read().splitlines()

# print(json_data[5])

utterance_list = []
trun_id_list = []
for idx, (utterance, turnID) in enumerate(
                        zip(utterance_list['utterance'], trun_id_list['turnID'])):
                        
                        print(utterance_list)
                        print(trun_id_list)
                        

# json_data = pd.DataFrame(for i in range(0, len(form_json_data)):
#                          json_data['data'][0]["body"]['dialogue'], 
#                          columns = ['utterance', 'utteranceID', 'participantID', 'date', 'turnID', 'time'])

# print(json_data)
# form_json_data= json_data.drop(['id', 'start', 'end','note','original_form'], axis=1)
# original_form_json_data= json_data.drop(['id', 'start', 'end','note','form'], axis=1)

# print(form_json_data.shape)
# print(original_form_json_data.shape)
                                   

# # print(json_data)

# # json_data.set_index('speaker_id', inplace = True)

# # print(json_data)

# # for form, original_form in zip(json_data['form'], json_data['original_form']):
# #     print(form, original_form)
#     # if text['speaker_id'] == text['speaker_id']: 

#     #     print(text)

#     #     tmp.append(json_data['form'], json_data['original_form'])

#     # print(tmp)

# for line in range(0, len(form_json_data)):
#     # for id in json_data['speaker_id']:
#     if form_json_data['speaker_id'][line] == form_json_data['speaker_id'][line+1]:
#         tmp = []
#         tmp.append(form_json_data['form'].values)

#     elif form_json_data['speaker_id'].empty != form_json_data['speaker_id'].empty:
#         break
# print(tmp)

# tmp = pd.DataFrame(tmp)

# print(tmp)
# # test = json_data['form'][0] + json_data['form'][1]
# # print(test)

## 2021년 6월 23~25일 제주도 KCC학회 참여 -> 내일 커밋 못할예정 