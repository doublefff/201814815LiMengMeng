import VSM_lemmated_50
import VSM_stemmed_50
import math
import os
import pandas as pd
import numpy as np

# tf*idf
def TF_IDF1(pp_documents,dictionary):
    vectors=[]
    if not os.path.exists('data/out/vsm_stemmed-tf-idf1.csv'):
        i=0
        for doc in pp_documents:
            vector=[]
            for item in dictionary:
                #计算TF
                # tf1=count(item) / sum(count.values())
                doc=" ".join(doc)
                tf=doc.count(item)
                if tf>0:
                    tf=1+math.log(tf)
                # 计算IDF
                df=len(list(filter(lambda doc: item in doc,pp_documents)))
                idf=math.log(len(pp_documents) / df)
                weight=tf*idf
                vector.append(weight)
            i+=1
            print(i)
            vectors.append(vector)
        pd.DataFrame(vectors).to_csv('data/out/vsm_stemmed-tf-idf1.csv', sep=",", header=None, index=None)
    else:
        vectors = np.array(pd.read_csv('data/out/vsm_stemmed-tf-idf1.csv', sep="\n", header=None))
    return vectors

if __name__ == '__main__':
    # 输入数据
    documents, labels = VSM_stemmed_50.input_data()
    #处理数据
    post_documents = VSM_stemmed_50.preprocessor()
    # 生成词典
    pp_documents,dictionary = VSM_stemmed_50.dict_create(post_documents)
    # tf-idf 获取vector space
    vectors = TF_IDF1(pp_documents, dictionary)



