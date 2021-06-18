import json # import json module
import pandas as pd 

# with statement
with open('data/SDRW2000000001.json') as json_file:
    json_data = json.load(json_file)

print(json_data.keys())

print(json_data['document'][0])

# json_data = pd.DataFrame(json_data['prizes'][0]["laureates"], 
#                         columns = ['id', 'firstname', 'surname', 'share', 'motivation'])

