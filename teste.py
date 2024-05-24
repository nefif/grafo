import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

# Função para realizar clustering e análise
def analyze_clusters(filepath, num_clusters):
    stop_words = list(ENGLISH_STOP_WORDS)

    # Leitura do dataset
    df_courses = pd.read_csv(filepath, delimiter=',', low_memory=False, encoding='UTF-8')
    df_courses = df_courses.dropna()
    df_courses = df_courses.drop_duplicates('Course Description')

    # Extrair as descrições dos cursos
    descriptions = df_courses['Course Description']

    # Calcular TF-IDF das descrições
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

    # Obter as palavras-chave mais relevantes para cada descrição
    terms = tfidf_vectorizer.get_feature_names_out()
    top_keywords_per_description = [
        [terms[idx] for idx in tfidf_matrix[i].toarray().flatten().argsort()[::-1][:6]]
        for i in range(len(descriptions))
    ]
    df_courses['Keywords'] = top_keywords_per_description

    # Realizar clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)
    df_courses['Cluster'] = labels

    # Identificar as 6 palavras-chave mais frequentes em cada cluster e armazenar os títulos
    cluster_titles = []
    for cluster in range(num_clusters):
        cluster_keywords = df_courses[df_courses['Cluster'] == cluster]['Keywords'].tolist()
        flattened_keywords = [keyword for sublist in cluster_keywords for keyword in sublist]
        keyword_counts = Counter(flattened_keywords)
        top_keywords = keyword_counts.most_common(6)
        top_keywords_str = ', '.join([keyword for keyword, count in top_keywords])
        title = f"Grupo de Cursos {cluster + 1} - {top_keywords_str}"
        cluster_titles.append(title)

    return df_courses, cluster_titles, kmeans, tfidf_matrix

# Função para visualizar os clusters usando PCA
def visualize_clusters(df_courses, pca_result, highlight_cluster=None):
    plt.figure(figsize=(10, 7))

    # Definindo as cores
    base_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']  # Até 10 clusters
    colors = ['gray' for _ in range(len(df_courses))]
    if highlight_cluster is not None:
        for idx in range(len(df_courses)):
            if df_courses.iloc[idx]['Cluster'] == highlight_cluster:
                colors[idx] = base_colors[highlight_cluster % len(base_colors)]

    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Visualização dos Clusters usando PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')


    st.pyplot(plt)

# Função para visualizar os gráficos de silhueta e do método do cotovelo
def visualize_evaluation(silhouette_scores, inertias, max_clusters):
    plt.figure(figsize=(15, 6))

    # Plotar o gráfico de silhueta
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    # Plotar o gráfico do método do cotovelo
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.title('Método do Cotovelo (Elbow Method)')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.grid(True)

    st.pyplot(plt)
    
# Função para construir o grafo para um cluster específico
def build_graph_for_cluster(cluster_courses, threshold=0.0):
    G = nx.DiGraph()
    
    # Construir matriz de similaridade entre keywords
    keyword_lists = cluster_courses['Keywords'].tolist()
    num_courses = len(keyword_lists)
    similarity_matrix = np.zeros((num_courses, num_courses))
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            keywords_i = set(keyword_lists[i])
            keywords_j = set(keyword_lists[j])
            similarity = len(keywords_i.intersection(keywords_j)) / len(keywords_i.union(keywords_j))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Adicionar arestas ao grafo com base no limiar de similaridade
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            if similarity_matrix[i, j] > threshold:
                course_i = cluster_courses.iloc[i]['Course Name']
                course_j = cluster_courses.iloc[j]['Course Name']
                G.add_edge(course_i, course_j, weight=similarity_matrix[i, j])
    
    return G

# Salvar grafos em arquivos .gexf
def save_graphs_as_gexf(df_courses, cluster_titles):
    for i, title in enumerate(cluster_titles):
        cluster_courses = df_courses[df_courses['Cluster'] == i][['Course Name', 'Keywords']]
        G = build_graph_for_cluster(cluster_courses)
        nx.write_gexf(G, f"{title}.gexf")
        
def show_graph_metrics(G):
    metrics = {}

    # Coeficiente de aglomeração médio
    clustering_coefficient = nx.average_clustering(G)
    metrics['Clustering Coefficient'] = clustering_coefficient

    # Centralidade de grau médio
    degree_centrality = nx.degree_centrality(G)
    avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    metrics['Average Degree Centrality'] = avg_degree_centrality
    max_degree_centrality_node = max(degree_centrality, key=degree_centrality.get)
    metrics['Nó com maior Centralidade é'] = max_degree_centrality_node

    # Centralidade de proximidade média
    closeness_centrality = nx.closeness_centrality(G)
    avg_closeness_centrality = sum(closeness_centrality.values()) / len(closeness_centrality)
    metrics['Average Closeness Centrality'] = avg_closeness_centrality
    max_closeness_centrality_node = max(closeness_centrality, key=closeness_centrality.get)
    metrics['Nó com a maior centralidade de fecho é'] = max_closeness_centrality_node


    # Centralidade de intermediação média
    betweenness_centrality = nx.betweenness_centrality(G)
    avg_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    metrics['Average Betweenness Centrality'] = avg_betweenness_centrality
    max_betweenness_centrality_node = max(betweenness_centrality, key=betweenness_centrality.get)
    metrics['Nó com a maior centralidade de intermediação é'] = max_betweenness_centrality_node

    # Densidade do grafo
    density = nx.density(G)
    metrics['Graph Density'] = density
    esparsity_threshold = 0.5
    is_sparse = density < esparsity_threshold
    metrics['O Grafo é Esparso'] = is_sparse

    # Assortatividade
    assortativity = nx.degree_assortativity_coefficient(G)
    metrics['Assortativity'] = assortativity

    # Diâmetro do grafo (apenas para grafos não direcionados)
    if not nx.is_directed(G):
        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            diameters = [nx.diameter(component) for component in nx.connected_components(G)]
            diameter = max(diameters)
        metrics['Graph Diameter'] = diameter

    # Excentricidade média (apenas para grafos não direcionados)
    if not nx.is_directed(G):
        eccentricity = nx.eccentricity(G)
        avg_eccentricity = sum(eccentricity.values()) / len(eccentricity)
        metrics['Average Eccentricity'] = avg_eccentricity

    # Assimetria direcional
    is_directed = nx.is_directed(G)
    metrics['Is Directed Graph'] = is_directed

    return metrics       
        
# Main application
def main():
    st.title("Análise de Clusters de Cursos")

    # Parâmetros da aplicação
    num_clusters = st.sidebar.slider('Número de Clusters', min_value=2, max_value=10, value=7)

    
    # Analisar clusters
    df_courses, cluster_titles, kmeans, tfidf_matrix = analyze_clusters('Coursera.csv', num_clusters)

    # Visualização dos clusters usando PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    
    # Calcular a silhueta e a inércia para 10 clusters
    silhouette_scores_10 = []
    inertias_10 = []
    for num_clusters_10 in range(2, 11):  # Renomeando a variável num_clusters dentro do loop
        kmeans = KMeans(n_clusters=num_clusters_10, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        silhouette_avg = silhouette_score(tfidf_matrix, labels)
        silhouette_scores_10.append(silhouette_avg)
        inertias_10.append(kmeans.inertia_)

    # Visualizar as avaliações
    with st.expander("Cotovolo / Silhueta "):
        visualize_evaluation(silhouette_scores_10, inertias_10, max_clusters=10)

    # Exibir títulos dos clusters
    for i, title in enumerate(cluster_titles):
        with st.expander(title):
            cluster_courses = df_courses[df_courses['Cluster'] == i][['Course Name', 'University', 'Course Description', 'Keywords']]
            st.dataframe(cluster_courses)

            # Mostrar análise do cluster
            st.subheader("Análise do Cluster")
            num_items = cluster_courses.shape[0]
            num_unique_universities = cluster_courses['University'].nunique()
            keyword_counts = Counter([keyword for sublist in cluster_courses['Keywords'] for keyword in sublist])

            st.write(f"Quantidade de Itens: {num_items}")
            st.write(f"Número diferente de Universidades: {num_unique_universities}")
            st.write("Contagem das 10 Keywords mais frequentes:")
            for keyword, count in keyword_counts.most_common(10):
                st.write(f"{keyword}: {count}")
                
            G = build_graph_for_cluster(cluster_courses)
            # Calcular e exibir as métricas do grafo
            metrics = show_graph_metrics(G)
            st.subheader("Métricas do Grafo:")
            st.write(metrics)
            nx.write_gexf(G, f"{title}.gexf")
            st.success(f"O grafo '{title}.gexf' foi salvo com sucesso")
            
            visualize_clusters(df_courses, pca_result, highlight_cluster=i)

if __name__ == "__main__":
    main()