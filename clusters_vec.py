import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.vocab import GloVe
import torchtext
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.neural_network import MLPRegressor
import os
from wordcloud import WordCloud


# Desativar o aviso de depreciação do torchtext
torchtext.disable_torchtext_deprecation_warning()

# Configurações iniciais
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Carregar os dados
df_des = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False, encoding='UTF-8')
df_des = df_des[['Course Name', 'University', 'Course Description']]
df_des = df_des.dropna()
df_des = df_des.drop_duplicates('Course Description')

""" n_sample = 300
# Reduzir o tamanho do dataframe para 300 amostras
df_des = df_des.sample(n=n_sample, random_state=42) """

# Inicializar o lematizador e o stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Função para lematizar as palavras-chave
def lemmatize_keywords(keywords):
    return [lemmatizer.lemmatize(word) for word in keywords]

# Função para stematizar as palavras-chave
def stem_keywords(keywords):
    return [stemmer.stem(word) for word in keywords]

# Função para extrair e pré-processar as principais palavras-chave
def preprocess_keywords(text, vectorizer, preprocessing_func=None, n_keywords=6):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = X.toarray().argsort()[0][::-1]
    keywords = [feature_names[i] for i in sorted_items[:n_keywords]]
    if preprocessing_func:
        keywords = preprocessing_func(keywords)
    keywords.sort()  # Ordenar as palavras-chave alfabeticamente
    return ' '.join(keywords)

# Configurar o TfidfVectorizer
stop_words = list(stopwords.words('english'))  # Convertendo para lista
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000)
tfidf_vectorizer.fit(df_des['Course Description'])

# Aplicar a função para cada descrição com lematização
df_des['Lemmatized Keywords'] = df_des['Course Description'].apply(lambda x: preprocess_keywords(x, tfidf_vectorizer, lemmatize_keywords))

# Aplicar a função para cada descrição com stematização
df_des['Stemmed Keywords'] = df_des['Course Description'].apply(lambda x: preprocess_keywords(x, tfidf_vectorizer, stem_keywords))

# Aplicar a função para cada descrição sem lematização e stematização
df_des['Original Keywords'] = df_des['Course Description'].apply(lambda x: preprocess_keywords(x, tfidf_vectorizer))

# Carregar embeddings GloVe
glove = GloVe(name='6B', dim=100)

# Função para obter o vetor GloVe de uma lista de palavras
def get_glove_vectors(words, glove_model):
    vectors = [glove_model[word] for word in words if word in glove_model.stoi]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(glove_model.dim)

# Aplicar a função para cada conjunto de palavras-chave lematizadas
df_des['GloVe Vectors'] = df_des['Lemmatized Keywords'].apply(lambda x: get_glove_vectors(x, glove))

# Converter a coluna de vetores GloVe para uma matriz numpy
X = np.vstack(df_des['GloVe Vectors'].values)

n_cluster = 6
# Aplicar o Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=n_cluster)
df_des['Cluster'] = clustering.fit_predict(X)

""" # Imprimir os clusters no console
for cluster_num in range(n_cluster):
    print(f"Cluster {cluster_num}:")
    cluster_courses = df_des[df_des['Cluster'] == cluster_num]
    for _, row in cluster_courses.iterrows():
        print(f"Course Name: {row['Course Name']}, University: {row['University']}, Lemmatized Keywords: {row['Lemmatized Keywords']}") """

# Salvar o resultado em um novo arquivo .csv para cada tipo de pré-processamento
df_des.to_csv('Clustered_Courses_Lemmatized.csv', index=False, columns=['Course Name', 'University', 'Lemmatized Keywords', 'Cluster'])
df_des.to_csv('Clustered_Courses_Stemmed.csv', index=False, columns=['Course Name', 'University', 'Stemmed Keywords', 'Cluster'])
df_des.to_csv('Clustered_Courses_Original.csv', index=False, columns=['Course Name', 'University', 'Original Keywords', 'Cluster'])

# Função para plotar gráficos de dispersão
def plot_scatter(X, labels, title, filename):
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.colorbar(label='Cluster')
    plt.savefig(filename)
    plt.close()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_scatter(X_pca, df_des['Cluster'], 'Gráfico de Dispersão dos Clusters (PCA)', 'scatter_clusters_pca_lem.png')

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=3000)
X_tsne = tsne.fit_transform(X)
plot_scatter(X_tsne, df_des['Cluster'], 'Gráfico de Dispersão dos Clusters (t-SNE)', 'scatter_clusters_tsne_lem.png')

# MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
plot_scatter(X_mds, df_des['Cluster'], 'Gráfico de Dispersão dos Clusters (MDS)', 'scatter_clusters_mds_lem.png')

# Autoencoder
autoencoder = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
autoencoder.fit(X, X)  # Usamos X como rótulos de saída também
X_autoencoder = autoencoder.predict(X)
plot_scatter(X_autoencoder, df_des['Cluster'], 'Gráfico de Dispersão dos Clusters (Autoencoder)', 'scatter_clusters_autoencoder_lem.png')

# Diretório onde os arquivos CSV serão salvos
output_dir = 'Clustered_Courses'

# Criar o diretório se não existir
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre cada cluster
for cluster_num in range(n_cluster):
    # Filtrar o DataFrame para o cluster atual
    cluster_df = df_des[df_des['Cluster'] == cluster_num]
    
    # Nome do arquivo CSV para este cluster
    csv_filename = os.path.join(output_dir, f'Cluster_{cluster_num}_Lem.csv')
    
    # Salvar o DataFrame correspondente a este cluster em um arquivo CSV
    cluster_df.to_csv(csv_filename, index=False, columns=['Course Name', 'University', 'Lemmatized Keywords', 'Cluster'])
    
# Função para criar e exibir uma Word Cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Gerar e exibir uma Word Cloud para cada cluster
for cluster_num in range(n_cluster):
    cluster_courses = df_des[df_des['Cluster'] == cluster_num]
    cluster_text = ' '.join(cluster_courses['Lemmatized Keywords'])
    generate_wordcloud(cluster_text, f'Word Cloud - Cluster {cluster_num}')
