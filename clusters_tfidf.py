import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import os

# Configurações iniciais
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') 


# Inicializar o lematizador
lemmatizer = WordNetLemmatizer()

# Função para lematizar palavras-chave
def lemmatize_keywords(keywords):
    return [lemmatizer.lemmatize(word) for word in keywords]

# Função para extrair as principais palavras-chave e lematizá-las
def extract_and_lemmatize_keywords(text, vectorizer, n_keywords=6):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = X.toarray().argsort()[0][::-1]
    keywords = [feature_names[i] for i in sorted_items[:n_keywords]]
    lemmatized_keywords = lemmatize_keywords(keywords)
    return ' '.join(lemmatized_keywords)

# Carregar os dados
df_courses = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False, encoding='UTF-8')
df_courses = df_courses[['Course Name', 'University', 'Course Description']]
df_courses = df_courses.dropna()
df_courses = df_courses.drop_duplicates('Course Description')
n_sample = 300
# Reduzir o tamanho do dataframe para 300 amostras
df_courses = df_courses.sample(n=n_sample, random_state=42)

# Inicializar o vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_courses['Course Description'])

# Lematizar as palavras-chave
df_courses['Lemmatized Keywords'] = df_courses['Course Description'].apply(lambda x: extract_and_lemmatize_keywords(x, tfidf_vectorizer))

# Realizar o Agglomerative Clustering
n_clusters = 6
clustering = AgglomerativeClustering(n_clusters=n_clusters)
cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())  # Converter para matriz densa

# Adicionar os rótulos dos clusters ao DataFrame
df_courses['Cluster'] = cluster_labels

# Gerar uma Word Cloud para cada cluster
for cluster_num in range(n_clusters):
    cluster_courses = df_courses[df_courses['Cluster'] == cluster_num]
    cluster_text = ' '.join(cluster_courses['Lemmatized Keywords'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - Cluster {cluster_num}')
    plt.savefig(f'wordcloud_cluster_{cluster_num}.png')

# Salvar um arquivo .csv para cada cluster gerado
output_dir = 'Clustered_Courses'
os.makedirs(output_dir, exist_ok=True)
for cluster_num in range(n_clusters):
    cluster_courses = df_courses[df_courses['Cluster'] == cluster_num]
    csv_filename = os.path.join(output_dir, f'Cluster_{cluster_num}_tfidf.csv')
    cluster_courses.to_csv(csv_filename, index=False)

# Reduzir a dimensionalidade para plotar o gráfico
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(tfidf_matrix.toarray())  # Converter para matriz densa

# Plotar o gráfico de dispersão dos clusters usando TSNE
plt.figure(figsize=(10, 7))
for cluster_num in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster_num)[0]
    plt.scatter(X_tsne[cluster_indices, 0], X_tsne[cluster_indices, 1], label=f'Cluster {cluster_num}', alpha=0.7)
plt.title('Gráfico de Dispersão dos Clusters (TSNE)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.savefig('scatter_clusters_tsne_tfidf.png')
plt.close()
