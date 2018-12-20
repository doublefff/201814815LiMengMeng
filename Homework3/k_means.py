import json
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer

max_df=0.5
min_df=2
max_features=1000


def read_data():
    documents=[]
    labels=[]
    with open("Tweets.txt",'r') as f:
        lines=f.read().split('\n')[:-1]
        for line in lines:
            line=json.loads(line)
            documents.append(line['text'])
            labels.append(line['cluster'])
    return documents,labels

def TFIDF(documents):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, use_idf=True,
                                 stop_words='english')  # tf-idf
    X = vectorizer.fit_transform(documents)
    return X.todense()

def score(labels, prediction):
    result_NMI = metrics.normalized_mutual_info_score(labels, prediction)
    print("result_NMI:", result_NMI)

def f_KMeans(X, Y, random_state):
    print('k-means')
    K = len(set(Y))
    prediction = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)
    score(Y, prediction)


def f_AffinityPropagation(X, Y):
    print('AffinityPropagation')
    prediction = AffinityPropagation().fit_predict(X)
    score(Y, prediction)


def f_MeanShift(X, Y):
    print('MeanShift')
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    prediction = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
    score(Y, prediction)


def f_SpectralClustering(X, Y, gamma):
    print('spectral_clustering')
    K = len(set(Y))
    prediction = SpectralClustering(n_clusters=K, gamma=gamma).fit_predict(X)
    score(Y, prediction)


def f_AgglomerativeClustering(X, Y, linkage):
    print('AgglomerativeClustering: ' + linkage)
    K = len(set(Y))
    prediction = AgglomerativeClustering(n_clusters=K, linkage=linkage).fit_predict(X)
    score(Y, prediction)


def f_DBSCAN(X, Y, eps, min_samples):
    print('DBSCAN')
    prediction = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    score(Y, prediction)


def f_GaussianMixture(X, Y, cov_type):
    print('GaussianMixture: ' + cov_type)
    K = len(set(Y))
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type).fit(X)
    prediction = gmm.predict(X)
    score(Y, prediction)

def main():
    documents,Y=read_data()
    X=TFIDF(documents)
    # K_Means
    f_KMeans(X, Y, random_state=13)
    # Affinity propagation
    f_AffinityPropagation(X, Y)
    # Mean-Shift
    f_MeanShift(X, Y)
    # SpectralClustering
    f_SpectralClustering(X, Y, gamma=0.06)
    # AgglomerativeClustering
    linkages = ['ward', 'average', 'complete']
    for linkage in linkages:
        f_AgglomerativeClustering(X, Y, linkage)
    # DBSCAN
    f_DBSCAN(X, Y, eps=0.3, min_samples=1)
    # GaussianMixture
    cov_types = ['spherical', 'diag', 'tied', 'full']
    for cov_type in cov_types:
        f_GaussianMixture(X, Y, cov_type)
if __name__ == '__main__':
    main()