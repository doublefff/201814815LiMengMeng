from DataLoader import dataloader
import numpy as np
from collections import Counter
import math

lb_name= ['alt.atheism','comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', \
        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', \
        'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', \
        'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', \
        'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']



def create_count(X_train,y_train,vocabSet):
    matrix=np.zeros((20,10131))
    prob_c_lt=np.zeros(20)
    for c in range(20):
        print(c)
        index_c=[]
        for i,x in enumerate(y_train):
            if x==lb_name[c]:
                print(i)
                index_c.append(i)

        print(index_c)
        num=len(index_c)
        prob_c=num/18828
        prob_c_lt[c]=prob_c

        list_sum=[]
        for i,x in enumerate(X_train):
            if i in index_c:
                while "" in x:
                    x.remove("")
                print(x)
                list_sum.extend(x)
        print(list_sum)
        print(len(list_sum))
        c_count=Counter(list_sum)

        print(c_count)

        j=0
        for voc in vocabSet:
            print(voc)
            if c_count[voc] !=None:
                matrix[c,j]=c_count[voc]
            j+=1
    np.save("matrix.npy",matrix)
    np.save("prob_c_lt.npy",prob_c_lt)
    return matrix,prob_c_lt

def trainNBO(matrix):
    p_matrix = np.zeros((20, 10131))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p_matrix[i,j]=matrix[i,j]/sum(matrix[i])

    return p_matrix

def testNBO(p_matrix,prob_c_lt,X_test,y_test,vocabSet):
    i=0
    count=0
    prob_lt=[]
    for doc in X_test:
        c_true=y_test[i]
        for c in range(20):
            for item in doc:
                if item in vocabSet:
                    inx=list(vocabSet).index(item)
                    if p_matrix[c][inx] !=0:
                        prob+=math.log(p_matrix[c][inx])
            prob=prob+math.log(prob_c_lt[c])
        prob_lt.append(prob)
        c_predict=prob_lt.index(max(prob_lt))
        i+=1
        if lb_name[c_predict]==c_true:
            count+=1
            print("正确"+count)
        print("文档"+i)
    accuracy=count/i

    return accuracy

if __name__ == '__main__':
    X_train, X_test, y_train, y_test,vocabSet=dataloader()
    print('1111')
    matrix,prob_c_lt=create_count(X_train,y_train,vocabSet)
    print("2222")
    p_matrix=trainNBO(matrix)
    print("3333")
    accuracy=testNBO(p_matrix,prob_c_lt,X_test,y_test,vocabSet)
    print("Accuracy is " + accuracy)