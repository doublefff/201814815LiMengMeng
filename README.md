# 201814815LiMengMeng

## 实验一

### 实验内容：

对英文数据集进行预处理、建立向量空间模型，然后使用KNN进行分类。


数据集：

20news-18828


实验步骤：


1、预处理

date:2018-10-18

完成字典的构建和TF-IDF值的计算以及文档的向量表示

在构建字典时采用了两种方式，一种是Stemming,一种是lemmatation,同时为了减少字典规模，对词频进行了过滤操作。


字典规模：

Size Stemmed Lemmated

5 33130 37675

20 10131 11728

50 5251 5948


2、KNN

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

在上一个实验中，我们使用knn算法来对文档进行分类，生成了词典，以及过滤后的文档，可以直接读取文件然后实现我们的朴素贝叶斯分类器。本次实验中一共有20 类，根据实验要求，我们需要统计每个类别所包含的单词以及该单词在本类中出现的概率。

2.基础模型

最大后验概率的计算

注意：（1）条件独立的假设

      （2）平滑处理：测试集中出现的词汇在训练集中没有出现
      
3.伯努利模型

在统计与判断时，将重复的词语视为其只出现1次

4.多项式模型

重复的词语视为其出现多次

5.混合模型

在计算句子概率时，不考虑重复词语出现的次数，但是在计算词语的概率时，却考虑重复词语的出现次数

6.平滑技术、取对数来转换权重

实验结果：

Accuracy: 0.80

实验三

1、实验任务

（1）测试sklearn中以下聚类算法在Tweet数据集上的聚类效果

（2）使用NMI（Normalized Mutual Information）作为评价指标


2、实验步骤

（1）K-Means

class sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)

（2）Affinity Propagation

class sklearn.cluster.AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity=’euclidean’, verbose=False)

（3）Mean-Shift

sklearn.cluster.mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, max_iter=300, n_jobs=None)

（4）Spectral Clustering

class sklearn.cluster.SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity=’rbf’, n_neighbors=10, eigen_tol=0.0, assign_labels=’kmeans’, degree=3, coef0=1, kernel_params=None, n_jobs=None)

（5）Agglomerative Clustering

class sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity=’euclidean’, memory=None, connectivity=None, compute_full_tree=’auto’, linkage=’ward’, pooling_func=’deprecated’)
linkage的参数选项有4种：ward、complete、average、single。在本次实验中，使用了3种参数，分别为ward、average、complete。

（6）DBSCAN

class sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric=’euclidean’, metric_params=None, algorithm=’auto’, leaf_size=30, p=None, n_jobs=None)

（7）Gaussian Mixtures

class sklearn.mixture.GaussianMixture(n_components=1, covariance_type=’full’, tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params=’kmeans’, weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10) 
covariance的参数选项有4种：spherical, diag, tied, full, 在本次实验中，使用了四种参数。

（8）评价指标NMI

sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method=’warn’)
用于评价聚类算法的性能

3、实验结果

聚类算法	NMI

K-Means	81.39%

Affinity Propagation	63.13%

Mean-Shift	78.66%

Spectral Clustering	77.67%


Agglomerative Clustering(Ward)	80.65%

Agglomerative Clustering(Average)	89.07%

Agglomerative Clustering(Complete)	71.91%

DBSCAN	70.20%

Gaussian Mixtures(spherical)	78.70%

Gaussian Mixtures(diag)	82.35%

Gaussian Mixtures(tied)	78.96%

Gaussian Mixtures(full)	74.33%
