import VSM_lemmated
import VSM_stemmed
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import numpy as np

def TF_IDF2(post_documents):
    if not os.path.exists('data/out/vsm_stemmed-tf-idf2.csv'):
        vectorizer = TfidfVectorizer(min_df=1)
        corpus = [str(doc) for doc in post_documents]
        vectorizer.fit_transform(corpus)
        dictionary = vectorizer.get_feature_names()
        vectors = vectorizer.fit_transform(corpus).toarray()
        pd.DataFrame(list(vectors)).to_csv('data/out/vsm_stemmed-ft-id2.csv', sep=",", header=None, index=None)
        pd.DataFrame(dictionary).to_csv('data/out/vsm_stemmed-ft-id2-dictionary.csv', sep=",", header=None, index=None)
    else:

        vectors = np.array(pd.read_csv('data/out/vsm_stemmed-ft-idf2.csv', sep="\n", header=None))
        dictionary=np.array(pd.read_csv('data/out/vsm_stemmed-ft-idf2-dictionary.csv', sep="\n", header=None))
    return dictionary, vectors

if __name__ == '__main__':
    # 输入数据
    documents, labels = VSM_stemmed.input_data()
    #处理数据
    post_documents = VSM_stemmed.preprocessor()
    # tf-idf 获取vector space
    dictionary,vectors = TF_IDF2(post_documents)