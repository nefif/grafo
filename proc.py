import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import text
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Adicionando as stopwords padrão do sklearn
stop_words = list(text.ENGLISH_STOP_WORDS)

# Grafo construído baseado na DESCRIÇÃO do curso
df_des = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False, encoding='UTF-8')
df_des = df_des[['Course Name', 'University', 'Course Description']]
df_des = df_des.dropna()
df_des = df_des.drop_duplicates('Course Description')

# Extrair as descriç�es dos cursos
descriptions = df_des['Course Description']

# Calcular TF-IDF das descriç�es
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Obter as palavras-chave mais relevantes para cada descrição
top_keywords_per_description = []
terms = tfidf_vectorizer.get_feature_names_out()
for i in range(len(descriptions)):
    top_indices = tfidf_matrix[i].toarray().argsort()[0][::-1][:6]
    top_keywords_per_description.append([terms[idx] for idx in top_indices])

# Adicionar a coluna 'Keywords' ao DataFrame df_des
df_des['Keywords'] = top_keywords_per_description

# Codificar as palavras-chave usando one-hot encoding
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
keywords_encoded = mlb.fit_transform(df_des['Keywords'])

# Aplicar o agrupamento hierárquico com Single Linkage
clustering_single = AgglomerativeClustering(linkage='single', n_clusters=4).fit(keywords_encoded)

# Aplicar o agrupamento hierárquico com Ward Linkage
clustering_ward = AgglomerativeClustering(linkage='ward', n_clusters=4).fit(keywords_encoded)

# Reduzir a dimensionalidade para 2D usando t-SNE
tsne = TSNE(n_components=2, random_state=0)
keywords_tsne = tsne.fit_transform(keywords_encoded)

# Plotar os clusters com Single Linkage
plt.figure(figsize=(10, 8))
scatter = plt.scatter(keywords_tsne[:, 0], keywords_tsne[:, 1], c=clustering_single.labels_)
plt.title('Visualização dos Clusters com Single Linkage')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.colorbar(scatter)
plt.show()

# Plotar os clusters com Ward Linkage
plt.figure(figsize=(10, 8))
scatter = plt.scatter(keywords_tsne[:, 0], keywords_tsne[:, 1], c=clustering_ward.labels_)
plt.title('Visualização dos Clusters com Ward Linkage')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.colorbar(scatter)
plt.show()