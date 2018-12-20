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

在过滤词频小于20的字典上完成KNN预测分类，数据集被分成训练集：测试集=8:2，对于每个测试集上的文档，计算它与训练集上文档的余弦相似度，返回K个最相似的预测它的类别，最后得到在整个数据集上的Accuarcy。

Stemmed-20:

1-based KNN Accuary is 0.8576739245884227

2-based KNN Accuary is 0.8576739245884227

3-based KNN Accuary is 0.8590015932023367

4-based KNN Accuary is 0.8600637280934679

5-based KNN Accuary is 0.8531598513011153

6-based KNN Accuary is 0.8613913967073819

7-based KNN Accuary is 0.8563462559745088

8-based KNN Accuary is 0.8544875199150292

9-based KNN Accuary is 0.8582049920339884

10-based KNN Accuary is 0.8417419012214551

11-based KNN Accuary is 0.8523632501327668

12-based KNN Accuary is 0.8499734466277217

13-based KNN Accuary is 0.8499734466277217

14-based KNN Accuary is 0.8454593733404142

15-based KNN Accuary is 0.8446627721720659

实验二

-Naive Bayes

时间：2018-11-25

实验步骤

1.实验预处理

在上一个实验中，我们使用knn算法来对文档进行分类，生成了词典，以及过滤后的文档，

可以直接读取文件然后实现我们的朴素贝叶斯分类器。本次实验中一共有20 类，根据实验要求，我们需要统计每个类别所包含的单词以及该单词在本类中出现的概率。
2.基础模型

最大后验概率的计算

注意：（1）条件独立的假设

      （2）平滑处理：测试集中出现的词汇在训练集中没有出现
      
3.伯努利模型

在统计与判断时，将重复的词语视为其只出现1次

4.多项式模型

重复的词语视为其出现多次

5.混合模型

在计算句子概率时 ，不考虑重复词语出现的次数 ，但是在统计 算词语的概率时 ，却考虑重复词语的出现次数

6.平滑技术、取对数来转换权重

实验结果：

Accuracy: 0.80
