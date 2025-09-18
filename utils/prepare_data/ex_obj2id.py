import json

data = []
with open('config/co_occur_freq.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
objs=set()
for line in data:
    objs.add(line['category1'])
    objs.add(line['category2'])
obj2id={}
for id,obj in enumerate(objs):
    obj2id[obj]=id
with open('config/object2id.json','w') as f:
    json.dump(obj2id,f)
    