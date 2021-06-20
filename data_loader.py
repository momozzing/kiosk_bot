import json # import json module
import pandas as pd 

# with statement
with open('data/SDRW2000000001.json') as json_file:
    json_data = json.load(json_file)

# print(json_data.keys())

# print(json_data['document'][0]['utterance'])

json_data = pd.DataFrame(json_data['document'][0]["utterance"], 
                         columns = ['id', 'form', 'original_form', 'speaker_id', 'start', 'end', 'note'])


json_data= json_data.drop(['id', 'start', 'end','note'], axis=1)

print(json_data)


for text in json_data:
    tmp = []
    if json_data[id] == json_data[id]: 

        tmp.append(json_data['form'], json_data['original_form'])

    print(tmp)