import json
import math
from collections import defaultdict
import numpy as np

input_file = "config/co_occur_freq.jsonl"
output_file = "config/co_occur_matrix.npy"
alpha = 0.5
obj2id_fp='config/object2id.json'
with open(obj2id_fp,'r') as f:
    obj2id=json.load(f)

pair_counts = defaultdict(int)
cat_counts = defaultdict(int)
categories = set()
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        c1, c2, count = obj2id[data["category1"]], obj2id[data["category2"]], data["count"]
        pair_counts[(c1, c2)] = count
        cat_counts[c1] += count
        cat_counts[c2] += count
        categories.update([c1, c2])

categories = sorted(list(categories))
V = len(categories)  
N = sum(pair_counts.values()) 

co_matrix=np.zeros((len(obj2id),len(obj2id)))

for i, c1 in enumerate(categories):
    for j, c2 in enumerate(categories):
        if i==j:
            co_matrix[c1][c2]=0
            continue
        
        count_ab=pair_counts.get((c1, c2),0)+1
        count_a=cat_counts[c1]+1
        count_b=cat_counts[c2]+1

        wr=math.log((count_ab*N)/(count_a*count_b),math.e)
        wn=-wr/math.log(count_ab/N,math.e)
        wc=(count_ab+1)/(count_ab+4)
        rs_score=(wn+1)*wc/2
        
        tr=count_ab/min(count_a,count_b)
        tc=max(0.3,1/(1+math.exp(20-min(count_a,count_b))))
        fs_score=tr*tc

        score = alpha * rs_score + (1 - alpha) * fs_score
        co_matrix[c1][c2] = round(score, 4)
        co_matrix[c2][c1] = round(score, 4)

max_val = np.max(co_matrix)
co_matrix=co_matrix/max_val
co_matrix = np.round(co_matrix, 4)
np.save(output_file,co_matrix)
