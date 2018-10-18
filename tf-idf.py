import VSM_lemmated
import VSM_stemmed
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import numpy as np

# tf*idf
def TF_IDF1(post_documents,dictionary):
    vectors=[]
    if not os.path.exists('data/out/vsm_stemmed-tf-idf1.csv'):
        i=0
        for doc in post_documents:
            vector=[]
            count=Counter(doc)
            for item in dictionary:
                #计算TF
                # tf1=count(item) / sum(count.values())
                tf=count(item)
                if tf>0:
                    tf=1+math.log(tf)
                # 计算IDF
                df=len(list(filter(lambda doc: item in doc,post_documents)))
                idf=math.log(len(post_documents) / df)
                weight=tf*idf
                vector.append(weight)
            i+=1
            print(i)
            vectors.append(vector)
        pd.DataFrame(vectors).to_csv('data/out/vsm_stemmed-ft-idf1.csv', sep=",", header=None, index=None)
    else:
        vectors = np.array(pd.read_csv('data/out/vsm_stemmed-ft-idf1.csv', sep="\n", header=None))
    return vectors

if __name__ == '__main__':
    # 输入数据
    documents, labels = VSM_stemmed.input_data()
    #处理数据
    post_documents = VSM_stemmed.preprocessor()
    # 生成词典
    dictionary = VSM_stemmed.dict_create(post_documents)
    # tf-idf 获取vector space
    vectors = TF_IDF1(post_documents, dictionary)



