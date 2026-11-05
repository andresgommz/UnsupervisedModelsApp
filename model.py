import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score

NUMERICAL = ['float64', 'int64']

def preprocessing(df, features):
    X = df[features].copy()

    numerical = X.select_dtypes(include=NUMERICAL).columns
    categorical = X.columns.difference(numerical)

    for col in numerical:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical:
        X[col] = X[col].fillna(X[col].mode()[0])

    scaler = StandardScaler()
    X[numerical] = scaler.fit_transform(X[numerical])
    encoder = LabelEncoder()
    X[categorical] = X[categorical].apply(encoder.fit_transform)

    return X

def reduce_pca(X, n_components):
    pca = PCA(n_components=n_components)
    cols = [f'PC{i + 1}' for i in range(n_components)]
    return pd.DataFrame(pca.fit_transform(X), columns=cols, index=X.index), sum(pca.explained_variance_ratio_)

def kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    model.fit(X)
    return model.labels_

def kmedoids(X, n_clusters, metric, max_samples=4000):
    model = KMedoids(n_clusters=n_clusters, metric=metric)
    model.fit(X if len(X) <= max_samples else X.sample(max_samples, random_state=42))
    return model.predict(X)

def dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model.labels_

def clustering(X, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(X)
    return model.labels_

def evaluate(X, labels):
    if not 2 <= len(set(labels)) < len(labels):
        return None, None
    return silhouette_score(X, labels), davies_bouldin_score(X, labels)