import os
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

b_path=r"./20news-18828"

#加载数据
def input_data():
    documents=[]
    labels=[]
    if not os.path.exists('data/output/documents.csv'):
        for folder in os.listdir(b_path):
            path=os.path.join(b_path,folder)
            for filename in os.listdir(path):
                filepath=os.path.join(path,filename)
                labels.append(folder)
                with open(filepath,'r',encoding="ISO-8859-1") as f:
                    document=f.read()
                    documents.append(document)
        docs=[str(doc) for doc in documents]
        pd.DataFrame(docs).to_csv('data/output/documents.csv', sep=" ", header=None, index=None)
        pd.DataFrame(labels).to_csv('data/output/labels.csv', sep=" ", header=None, index=None)
    else:
        labels = np.array(pd.read_csv('data/output/labels.csv', sep=" ", header=None))
        documents = np.array(pd.read_csv('data/output/documents.csv', sep=" ", header=None))

    return documents,labels

def preprocessor():
    post_documents=[]
    i=0
    if not os.path.exists('data/output/post_documents_stemmed.csv'):
        print("Processing")
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

            post_documents.append(stemmed)

        pd.DataFrame(post_documents).to_csv('data/output/post_documents_stemmed.csv', sep=" ", header=None, index=None)
    else:
        print("load post_documents")
        post_documents=np.array(pd.read_csv('data/output/post_documents_stemmed.csv', keep_default_na=False,sep=" ", header=None))
    return post_documents


# 返回字典
def dict_create(post_documents):
    pp_documents=[]
    dictionary=set()
    df={}
    if not os.path.exists('data/output/dictionary_stemmed_20.csv'):
        # 过滤词频
        count = Counter([token for doc in post_documents for token in doc])
        j = 0
        print("Filtering")
        for stemmed in post_documents:
            back = list(filter(lambda token: count[token] >= 20, stemmed))
            if "" in back:
                back.remove("")
            j += 1
            print(j)
            count1=dict(Counter(back))
            pp_documents.append(count1)
            for ik in count1.keys():
                if ik != "":
                   dictionary.add(ik)
                   if ik not in df:
                      df[ik]=1
                   else:
                      df[ik]+=1
        with open('data/output/df_stemmed_20.txt','w',encoding='utf-8') as f:
            f.write(str(df))
        with open('data/output/pp_documents_stemmed_20.txt','w',encoding='utf-8') as f:
            for line in pp_documents:
                 f.write(str(line)+"\n")

        print("Dictionary")
        print(len(dictionary))
        print(len(df))
        pd.DataFrame(list(dictionary)).to_csv('data/output/dictionary_stemmed_20.csv', sep=" ", header=None, index=None)
    else:
        print("load dictionary")
        with open('data/output/df_stemmed_20.txt','r',encoding='utf-8') as f:
            df=f.read()
            df=eval(df)

        with open('data/output/pp_documents_stemmed_20.txt','r',encoding='utf-8') as f:
            pp_documents=[]
            lines=f.readlines()
            for line in lines:
               pp_documents.append(eval(line))

        dictionary = np.array(pd.read_csv('data/output/dictionary_stemmed_20.csv', keep_default_na=False, sep=" ", header=None)).reshape(1,-1)[0]
    return df,pp_documents,dictionary


def vsm(post_documents,dictionary):
    vectors = []
    if not os.path.exists('data/output/vsm-01_stemmed_20.csv'):
        print("Vector Space model")
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
        pd.DataFrame(vectors).to_csv('data/output/vsm-01_stemmed_20.csv', sep=",", header=None, index=None)
    else:
        print("Exsiting 0-1 dictionary")
        return True
if __name__ == '__main__':
    # 输入数据
    documents, labels = input_data()
    # 处理数据
    post_documents = preprocessor()
    # 生成字典
    df,pp_documents,dictionary = dict_create(post_documents)
    # # 生成vector space
    # vsm(pp_documents, dictionary)

