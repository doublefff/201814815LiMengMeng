import VSM_20
import math
import os
import pandas as pd
import numpy as np
from collections import Counter

# tf*idf
def TF_IDF1(df,pp_documents,dictionary):
    vectors=[]
    if not os.path.exists('data/output/vsm_stemmed-tf-idf1-20.npy'):
        i=0
        for doc in pp_documents:
            vector=[]
            for item in dictionary:
                #计算TF
                # tf1=count(item) / sum(count.values())
                if item in doc:
                  tf=doc[item]
                else:
                  tf=0
                if tf>0:
                    tf=1+math.log(tf)
                # 计算IDF
                df_=df[item]
                idf=math.log(len(pp_documents) / df_)
                weight=tf*idf
                vector.append(weight)

            i+=1
            print(i)
            vectors.append(vector)
        np.save('data/output/vsm_stemmed-tf-idf1-20.npy',np.array(vectors))
    else:
        vectors=np.load('data/output/vsm_stemmed-tf-idf1-20.npy')
    return vectors

if __name__ == '__main__':
    # 输入数据
    documents, labels = VSM_20.input_data()
    #处理数据
    post_documents = VSM_20.preprocessor()
    # 生成词典
    df,pp_documents,dictionary = VSM_20.dict_create(post_documents)
    # tf-idf 获取vector space
    vectors = TF_IDF1(df,pp_documents, dictionary)



