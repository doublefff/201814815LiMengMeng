import os
import math
import numpy as np
import pandas as pd
import random

file = "data/output/vsm_stemmed-tf-idf1-20.npy"
vectors = np.load(file)
labels = np.array(pd.read_csv('data/output/labels.csv', sep=" ", header=None))
labels_lt = []
dict = {'alt.atheism': 1, 'comp.graphics': 2, 'comp.os.ms-windows.misc': 3, 'comp.sys.ibm.pc.hardware': 4, \
        'comp.sys.mac.hardware': 5, 'comp.windows.x': 6, 'misc.forsale': 7, 'rec.autos': 8, 'rec.motorcycles': 9, \
        'rec.sport.baseball': 10, 'rec.sport.hockey': 11, 'sci.crypt': 12, 'sci.electronics': 13, \
        'sci.med': 14, 'sci.space': 15, 'soc.religion.christian': 16, 'talk.politics.guns': 17, \
        'talk.politics.mideast': 18, 'talk.politics.misc': 19, 'talk.religion.misc': 20}
for j in range(18828):
    item = labels[j][0]
    labels_lt.append(dict[item])

list1 = range(0, 18828)
train_index = random.sample(list1, int(0.8 * len(list1)))
test_index = [item for item in list1 if item not in train_index]


def computeSim(v1, v2):
    num = np.matmul(v1, v2)
    num = float(num)
    denom_v1 = np.linalg.norm(v1)
    denom_v2 = np.linalg.norm(v2)
    sim = num / (denom_v1 * denom_v2)
    sim = 0.5 + 0.5 * sim
    return sim


def similarity_compute(v1, train_index):
    dic = {}
    for itr in train_index:
        v_ = vectors[itr]
        sim_ = computeSim(v1, v_)
        dic[itr] = sim_
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return dic


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def KNN(k):
    print("K is " + str(k))
    j = 0
    num = 0
    for ite in test_index:
        v1 = vectors[ite]
        dic = similarity_compute(v1, train_index)
        list = dic[:k]
        index = []
        for tu in list:
            index.append(tu[0])
        print(index)
        labels_k = []
        for s in index:
            labels_k.append(labels_lt[s])
        print(labels_k)
        max_str = max_list(labels_k)
        if max_str == labels_lt[ite]:
            num += 1
        j += 1
        print("test: " + str(j))
        print("num: " + str(num))
        acc_j = float(num / j)
        print("acc " + str(acc_j))
        with open('data/output/vsm_stemmed-tf-idf1-20-acc.txt', 'a', encoding='utf-8') as f:
            f.write(str(j) + "-test acc is " + str(acc_j) + "\n")
    acc = num / len(test_index)
    print(str(k) + " -based KNN Accuary is " + str(acc))
    with open('data/output/vsm_stemmed-tf-idf1-20-acc.txt', 'a', encoding='utf-8') as f:
        f.write(str(k) + "-based KNN Accuary is " + str(acc) + "\n")


if __name__ == '__main__':
    for k in range(1, 50):
        KNN(k)
