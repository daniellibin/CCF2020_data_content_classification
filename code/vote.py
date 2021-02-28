import re
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

directory = './result/'
files = [file for a, b, file in os.walk(directory)][0]
flag = False


def vote(predictions):
    '''
    投票融合方法
    :param predictions:
    :return:
    '''

    result = []
    for tmp in (predictions):
        counter = Counter(tmp)
        result.append(counter.most_common()[0][0])
    return result


res = [[] for _ in range(20000)]
print(res)

for file in files:
    tmp = pd.read_csv(directory + file)
    label = tmp["class_label"].to_list()
    for i in range(len(label)):
        res[i].append(label[i])


result = vote(res)

sub = pd.read_csv('data/submit_example.csv')
sub['class_label'] = result
def rank_label(label):
    if label in ["财经","时政"]:
        return "高风险"
    elif label in ["房产", "科技"]:
        return "中风险"
    elif label in ["教育", "时尚", "游戏"]:
        return "低风险"
    elif label in ["家居", "体育", "娱乐"]:
        return "可公开"

sub['rank_label'] = sub['class_label'].apply(rank_label)
sub.to_csv("./result——vote.csv", index=False)






