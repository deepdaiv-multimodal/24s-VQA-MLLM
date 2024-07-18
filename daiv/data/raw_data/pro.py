import json
from typing import Iterable

#annotation
annpath='/home/intern24/BLIVA/daiv/data/raw_data/mscoco_train2014_annotations.json'
with open(annpath, "r") as f:
    data = json.load(f)

#question
annpath2='/home/intern24/BLIVA/daiv/data/raw_data/OpenEnded_mscoco_train2014_questions.json'
with open(annpath2, "r") as f:
    data2 = json.load(f)

print(data['annotations'][100]['question_id'])
print(data2['questions'][100])
print(len(data['annotations']))
print(len(data2['questions']))
#data['annotations'][0]['questions']=data2['questions'][0]['question']
#print(data['annotations'][0])

#매핑
for i in range(len(data['annotations'])):
    data['annotations'][i]['question']=data2['questions'][i]['question']

#with open('BLIVA/daiv/data/raw_data/okvqa_train.json', 'w') as f:
#    json.dump(data,f)