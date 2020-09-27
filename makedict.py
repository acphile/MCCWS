import utils
import pickle
import os
from utils import is_dataset_tag, make_sure_path_exists

path="data/joint-sighan-simp/raw/train-all.txt"

out_path="dict.pkl"

dic={}
tokens={}
with open(path, "r", encoding="utf-16") as f:
    for line in f.readlines():
        cur=line.strip().split(" ")
        name=cur[0][1:-1]
        if dic.get(name) is None:
            dic[name]=set()
            tokens[name]=0
        tokens[name]+=len(cur[1:-1])
        dic[name].update(cur[1:-1])

for i in list(dic.keys()):
    print(i,len(dic[i]),tokens[i])
with open(out_path,"wb") as outfile:
    pickle.dump(dic,outfile)
