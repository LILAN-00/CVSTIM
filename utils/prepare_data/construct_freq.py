import json
from pycocotools.coco import COCO
from itertools import combinations

ann_file = "dataset/annotations/instances_train2014.json"
coco = COCO(ann_file)
img_ids = coco.getImgIds()
cooccur = {}
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    cat_ids = set(ann["category_id"] for ann in anns)
    for pair in combinations(sorted(cat_ids), 2):
        key = frozenset(pair)
        cooccur[key] = cooccur.get(key, 0) + 1

cats = coco.loadCats(coco.getCatIds())
id2name = {cat["id"]: cat["name"] for cat in cats}
all_cat_ids = sorted(id2name.keys())

all_pairs = [frozenset(pair) for pair in combinations(all_cat_ids, 2)]

results = []
for pair in all_pairs:
    id1, id2 = sorted(pair)
    count = cooccur.get(pair, 0)
    results.append({
        "category1": id2name[id1],
        "category2": id2name[id2],
        "count": count
    })

results = sorted(results, key=lambda x: x["count"], reverse=True)

with open("config/co_occur_freq.jsonl", "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")
