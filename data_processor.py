import os
from textblob import TextBlob
from textblob import Word

def data_load(path):

    index=0
    files=os.listdir(path)
    dic_t=dict()
    for file in files:
        list=[]
        file_list=os.listdir(path+"./"+file)
        for file_name in file_list:
            with open(path+"/"+file+"/"+file_name,'r',encoding="ISO-8859-1") as f:
                data=f.readlines()
                index+=1
                list.append(data)

        dic_t[file]=list

    return dic_t

def data_split(data):
    for key in data.keys():
        data_cur=data[key]
        sentence = TextBlob(str(data_cur).replace("\\\\n", "").replace("\\\\t", "").replace("\\", "").replace("'", ""))
        data[key]=sentence.words

    return data

def data_create(data,frequency):
    s=set()
    dic=dict()
    for key in data.keys():
        data_cur=data[key]
        for text in data_cur:
            w=Word(text)
            w.lemmatize()
            w.lemmatize('v')
            if dic.get(w) is None:
                dic[w]=1
            else:
                dic[w]+=1
            if w not in s and dic[w]>=frequency:
                s.add(w)

    return s

