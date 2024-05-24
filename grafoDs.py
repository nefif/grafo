import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.feature_extraction import text

# Adicionando as stopwords padrão do sklearn
stop_words = list(text.ENGLISH_STOP_WORDS)

# Grafo construído baseado na DESCRIÇÃO do curso
df_des = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False, encoding='UTF-8')
df_des = df_des[['Course Name', 'University', 'Course Description']]
df_des = df_des.dropna()
df_des = df_des.drop_duplicates('Course Description')

# Extrair as descrições dos cursos
descriptions = df_des['Course Description']

# Calcular TF-IDF das descrições
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

# Criar um novo vetorizador TF-IDF usando todas as palavras-chave relevantes
all_top_keywords = set(keyword for sublist in top_keywords_per_description for keyword in sublist)
new_tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, vocabulary=all_top_keywords)

# Recalcular a matriz TF-IDF usando apenas as palavras-chave relevantes
new_tfidf_matrix = new_tfidf_vectorizer.fit_transform(descriptions)

# Calcular a similaridade de cosseno entre as descrições usando a nova matriz TF-IDF
similarity_matrix = cosine_similarity(new_tfidf_matrix, new_tfidf_matrix)

# Construção do Grafo
G = nx.DiGraph()
num_courses = len(df_des)

limiar_similaridade = 0.5

for i in range(num_courses):
    for j in range(i + 1, num_courses):
        if similarity_matrix[i][j] > limiar_similaridade:
            course_i = df_des.iloc[i]['Course Name']
            course_j = df_des.iloc[j]['Course Name']
            G.add_edge(course_i, course_j, weight=similarity_matrix[i][j])

# Número de nós
num_nodes = G.number_of_nodes()
print("Número de nós:", num_nodes)

# Número de arestas
num_edges = G.number_of_edges()
print("Número de arestas:", num_edges)

# Salvar o grafo em um arquivo
nx.write_gexf(G, "cursos_similares_desc.gexf" )


# Métricas de rede
average_clustering_coefficient = nx.average_clustering(G)
assortativity_coefficient = nx.degree_assortativity_coefficient(G)
if nx.is_strongly_connected(G):
    average_distance = nx.average_shortest_path_length(G)
else:
    average_distance = None


# Ajustar valores e converter para porcentagem, se necessário
average_clustering_coefficient = round(average_clustering_coefficient, 4)
assortativity_coefficient = round(assortativity_coefficient, 4) * 100  # Convertendo para porcentagem


print("Coeficiente de Agrupamento Médio:", average_clustering_coefficient)
print("Coeficiente de Assortatividade:", assortativity_coefficient)
print("Distância Média:", average_distance)

selected_courses = [
    "Retrieve Data using Single-Table SQL Queries",
    "Silicon Thin Film Solar Cells"
]

for course in selected_courses:
    course_subgraph = G.subgraph(nx.single_source_shortest_path_length(G, course, cutoff=1))
    
    # Salvar o subgrafo em um arquivo .gexf
    nx.write_gexf(course_subgraph, f"{course}_related_courses.gexf")
    
    # Exibir as palavras-chave e o peso das arestas para cada curso
    print(f"Curso: {course}")
    for edge in course_subgraph.edges(data=True):
        source = edge[0]
        target = edge[1]
        weight = edge[2]['weight']
        keywords_source = df_des[df_des['Course Name'] == source]['Keywords'].iloc[0]
        keywords_target = df_des[df_des['Course Name'] == target]['Keywords'].iloc[0]
        print(f"  - Conexão entre {source} e {target}, peso: {weight}")
        print(f"    Palavras-chave de {source}: {keywords_source}")
        print(f"    Palavras-chave de {target}: {keywords_target}")
    print()