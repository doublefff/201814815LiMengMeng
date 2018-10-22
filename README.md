# 201814815LiMengMeng
COURSR:Data Mining 
HOMEWORK AND PROJECT:

-Vector space model

date:2018-10-18

完成字典的构建和TF-IDF值的计算以及文档的向量表示

在构建字典时采用了两种方式，一种是Stemming,一种是lemmatation,同时为了减少字典规模，对词频进行了过滤操作

下面是构建字典进行对比试验的结果

Size   Stemmed    Lemmated

5      33130      37675

20     10131      11728

50     5251       5948

date:2018-10-20

在过滤词频小于50的字典上完成KNN预测分类，数据集被分成训练集：测试集=8:2，对于每个测试集上的文档，计算它与训练集上文档的余弦相似度，返回K个最相似的预测它的类别，最后得到在整个数据集上的Accuarcy。

Stemmed-20:

Lemmated-20:

 
