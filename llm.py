import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import sys

# Aumentar o limite de recursão
sys.setrecursionlimit(10000)

nltk.download('stopwords')

# Carregar os dados
df_des = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False, encoding='UTF-8')
df_des = df_des[['Course Name', 'University', 'Course Description']]
df_des = df_des.dropna()
df_des = df_des.drop_duplicates('Course Description')

# Reduzir o tamanho do dataframe para 300 amostras
df_des = df_des.sample(n=300, random_state=42)

# Função para extrair as principais palavras-chave
def extract_keywords(text, vectorizer, n_keywords=6):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = X.toarray().argsort()[0][::-1]
    keywords = [feature_names[i] for i in sorted_items[:n_keywords]]
    return keywords

# Configurar o TfidfVectorizer uma vez
stop_words = list(stopwords.words('english'))  # Convertendo para lista
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
tfidf_vectorizer.fit(df_des['Course Description'])

# Aplicar a função para cada descrição
df_des['Keywords'] = df_des['Course Description'].apply(lambda x: extract_keywords(x, tfidf_vectorizer))

# Transformar as descrições em uma matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.transform(df_des['Course Description'])

# Reduzir a dimensionalidade para visualização usando SVD
svd = TruncatedSVD(n_components=50)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)
tfidf_matrix_reduced = normalize(tfidf_matrix_reduced)

# Método de linkage e dendrogramas
methods = ['ward', 'single']
for method in methods:
    Z = linkage(tfidf_matrix_reduced, method=method)
    plt.figure(figsize=(10, 7))
    plt.title(f'Dendrogram - {method} linkage')
    dendrogram(Z)
    plt.xlabel('Courses')
    plt.ylabel('Distance')
    plt.show()

# Função para calcular a inércia (necessária para o método do cotovelo)
def calculate_inertia(data, n_clusters):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clustering_model.fit(data)
    labels = clustering_model.labels_
    return sum([((data[labels == i] - data[labels == i].mean(axis=0)) ** 2).sum() for i in range(n_clusters)])

# Método do cotovelo
inertias = []
range_n_clusters = range(2, 11)
for n_clusters in range_n_clusters:
    inertia = calculate_inertia(tfidf_matrix_reduced, n_clusters)
    inertias.append(inertia)

plt.figure(figsize=(10, 7))
plt.plot(range_n_clusters, inertias, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Análise da silhueta e índice de Davies-Bouldin
silhouette_scores = []
davies_bouldin_scores = []

for n_clusters in range_n_clusters:
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering_model.fit_predict(tfidf_matrix_reduced)
    
    silhouette_avg = silhouette_score(tfidf_matrix_reduced, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    davies_bouldin_avg = davies_bouldin_score(tfidf_matrix_reduced, cluster_labels)
    davies_bouldin_scores.append(davies_bouldin_avg)

plt.figure(figsize=(10, 7))
plt.plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
plt.plot(range_n_clusters, davies_bouldin_scores, marker='s', label='Davies-Bouldin Score')
plt.title('Silhouette and Davies-Bouldin Scores For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.legend()
plt.show()

# Escolher o número ideal de clusters com base nas análises anteriores
optimal_n_clusters = 6  # Isso pode ser ajustado com base nas análises anteriores

# Clusterização usando Aglomerative Clustering
clustering_model = AgglomerativeClustering(n_clusters=optimal_n_clusters, metric='euclidean', linkage='ward')
df_des['Cluster'] = clustering_model.fit_predict(tfidf_matrix_reduced)

# Função para obter as principais palavras-chave de um cluster
def get_top_keywords(cluster_data, vectorizer, n_keywords=6):
    all_text = " ".join(cluster_data)
    X = vectorizer.transform([all_text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = X.toarray().argsort()[0][::-1]
    keywords = [feature_names[i] for i in sorted_items[:n_keywords]]
    return keywords

# Imprimir os cursos e palavras-chave de cada cluster
for cluster in range(optimal_n_clusters):
    cluster_data = df_des[df_des['Cluster'] == cluster]
    cluster_descriptions = cluster_data['Course Description'].tolist()
    top_keywords = get_top_keywords(cluster_descriptions, tfidf_vectorizer, n_keywords=6)
    print(f"\nCluster {cluster}:")
    print(f"Top Keywords: {', '.join(top_keywords)}")
    print(cluster_data[['Course Name', 'University']].to_string(index=False))

