import os
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
b_path=r"./20news-18828"

#加载数据
def input_data():
    documents=[]
    labels=[]
    if not os.path.exists('data/out/documents.csv'):
        for folder in os.listdir(b_path):
            path=os.path.join(b_path,folder)
            for filename in os.listdir(path):
                filepath=os.path.join(path,filename)
                labels.append(folder)
                with open(filepath,'r',encoding="ISO-8859-1") as f:
                    document=f.read()
                    documents.append(document)
        docs=[str(doc) for doc in documents]
        pd.DataFrame(docs).to_csv('data/out/documents.csv', sep=" ", header=None, index=None)
        pd.DataFrame(labels).to_csv('data/out/labels.csv', sep=" ", header=None, index=None)
    else:
        labels = np.array(pd.read_csv('data/out/labels.csv', sep=" ", header=None))
        documents = np.array(pd.read_csv('data/out/documents.csv', sep=" ", header=None))

    return documents,labels

#预处理
def preprocessor():
    post_documents=[]
    i=0
    if not os.path.exists('data/out/post_documents_stemmed_5.csv'):
        for doc in documents:
            doc = doc[0].replace("\n", "").replace("\r", "").replace("\t","")
            # Normalization
            lowers=str(doc).lower()
            # 去除特殊字符
            remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
            no_punctuation = lowers.translate(remove_punctuation_map)
            # Tokenization
            tokens = nltk.word_tokenize(no_punctuation)
            # Stopwords
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            # Stemming
            stemmer = PorterStemmer()
            stemmed = []
            for item in filtered:
                stemmed.append(stemmer.stem(item))
            i+=1
            print(i)

            # 过滤词频
            back = list(filter(lambda token: str(stemmed).count(token) >= 5, stemmed))
            post_documents.append(back)
            docs = [str(doc) for doc in post_documents]
            pd.DataFrame(docs).to_csv('data/out/post_documents_stemmed_5.csv', sep=" ", header=None, index=None)
    else:
        post_documents = np.array(pd.read_csv('data/out/post_documents_stemmed_5.csv', sep=" ", header=None))
    return post_documents

# 返回字典
def dict_create(post_documents):
    s=set()
    if not os.path.exists('data/out/dictionary_stemmed_5.csv'):
        i=0
        for doc in post_documents:
            for w in doc:
                s.add(w)
            i+=1
            print(i)
        print(len(s))
        pd.DataFrame(list(s)).to_csv('data/out/dictionary_stemmed_5.csv', sep=" ", header=None, index=None)
    else:
        s = np.array(pd.read_csv('data/out/dictionary_stemmed_5.csv', sep=" ", header=None)).reshape(1, -1)[0]
        print(len(s))
    return s

def vsm(post_documents,dictionary):
    print("vector space")
    vectors=[]
    i=0
    for document in post_documents:
        vector=[]
        for item in dictionary:
            if item in document:
                vector.append('1')
            else:
                vector.append('0')
        vectors.append(vector)
        i+=1
        print(i)
    pd.DataFrame(vectors).to_csv('data/out/vsm-01_stemmed_5.csv', sep=",", header=None, index=None)

if __name__ == '__main__':
    #输入数据
    documents,labels=input_data()
    #处理数据
    post_documents=preprocessor()
    #生成字典
    dictionary=dict_create(post_documents)
    # 生成vector space
    vsm(post_documents, dictionary)
    #
    # dictionary=dict_create(post_documents)
    # 1 308271
    # dictionary=dict_create(post_documents)
    # 10 17652