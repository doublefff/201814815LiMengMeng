import data_processor
from textblob import TextBlob

file="./20news-18828"
frequency=1

def build_dic(path):
    dic_step1=data_processor.data_load(path)
    dic_step2=data_processor.data_split(dic_step1)
    dic_step3=data_processor.data_create(dic_step2,frequency)
    print(len(dic_step3))
    with open("./dict "+str(frequency)+".txt","w+",encoding="utf-8") as f:
        for i in dic_step3:
            f.write(str(i)+"\n")

build_dic(file)

