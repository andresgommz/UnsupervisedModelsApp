import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import NUMERICAL, preprocessing, reduce_pca, kmeans, kmedoids, clustering, dbscan, evaluate

st.title('Unsupervised Models App')

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

mode = st.radio('Select feature mode', ['Manual Selection', 'PCA Reduction'])
columns = df.columns.to_list()
features = st.multiselect('Select the features for clustering', columns, default=columns if mode == 'PCA Reduction' else None)

if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

X = preprocessing(df, features)

if mode == 'PCA Reduction':
    numerical = df[features].select_dtypes(include=NUMERICAL).columns
    if len(numerical) < 2:
        st.warning('PCA needs at least 2 numerical features')
        st.stop()
    n_components = st.number_input('Number of components', 2, len(numerical), 2)
    X, explained = reduce_pca(X[numerical], n_components)
    st.write('Explained variance ratio: ', explained)

modelSelection = st.selectbox('Select model', ['K-Means', 'K-Medoids', 'DBSCAN', 'Hierarchical Clustering'])

if modelSelection == 'K-Means':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    labels = kmeans(X, n_clusters)

elif modelSelection == 'K-Medoids':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    metric = st.selectbox('Select metric', ['euclidean', 'manhattan', 'cosine'], index=1)
    labels = kmedoids(X, n_clusters, metric)

elif modelSelection == 'DBSCAN':
    eps = st.slider('Eps', 0.1, 5.0, 0.5, 0.05)
    min_samples = st.slider('Number of minimum points', 2, 20, 5)
    labels = dbscan(X, eps=eps, min_samples=min_samples)

else:
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    linkage = st.selectbox('Select linkage', ['ward', 'single', 'complete'])
    labels = clustering(X, n_clusters=n_clusters, linkage=linkage)

st.subheader('Clustered dataset')
clustered = df.copy()
clustered['cluster'] = labels
st.dataframe(clustered)

st.subheader('Evaluation')
silhouette, davies = evaluate(X, labels)
if silhouette is None:
    st.warning('The metrics need at least 2 distinct labels.')
else:
    st.write('Silhouette Score: ', silhouette)
    st.write('Davies-Bouldin Score: ', davies)

st.subheader('Visualization')
fig = plt.figure()
if X.shape[1] == 2:
    ax = fig.add_subplot(111)
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
else:
    P = X if mode == 'PCA Reduction' else reduce_pca(X, 3)[0]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P.iloc[:, 0], P.iloc[:, 1], P.iloc[:, 2], c=labels, cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
st.pyplot(fig)
