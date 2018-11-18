import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

def dataloader():
    if not os.path.exists('documents.csv'):
        post_documents = np.array(
            pd.read_csv('post_documents.csv', keep_default_na=False, sep=" ", header=None))

        documents=[]
        count = Counter([token for doc in post_documents for token in doc])
        for doc in post_documents:
            back = list(filter(lambda token: count[token] >= 20, doc))
            while "" in back:
                back.remove("")
            print(back)
            documents.append(back)


        pd.DataFrame(documents).to_csv('documents.csv', sep=" ", header=None, index=None)
    else:
        documents=np.array(pd.read_csv('documents.csv', sep=" ", header=None))

    labels = np.array(pd.read_csv('labels.csv', sep=" ", header=None))
    dictionary = np.array(pd.read_csv('dictionary.csv', sep=" ", header=None))

    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=0)

    return X_train,X_test,y_train,y_test,dictionary

if __name__ == '__main__':
    dataloader()
