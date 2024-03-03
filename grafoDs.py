import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Função para calcular a Distância de Jaccard entre dois conjuntos de palavras
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union if union != 0 else 0

df_des = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False)
df_des = df_des[['Course Name', 'University', 'Course Description']]
colunas_plt_ordem = ['University', 'Course Name', 'Course Description']
colunas_plt_nome = {'University':'Instituição', 'Course Name':'Título', 'Course Description':'Descrição'}
df_des = df_des.reindex(columns=colunas_plt_ordem)
df_des = df_des.rename(columns=colunas_plt_nome)
df_des['Instituição | Curso'] = df_des['Instituição'].astype(str) + ' - ' + df_des['Título'].astype(str)
df_des = df_des.dropna()
# Extrair as descrições dos cursos
descriptions = df_des['Descrição']

# Calcular TF-IDF das descrições
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Calcular a similaridade de cosseno entre as descrições
similarity_matrix_cosine = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Calcular a Distância de Jaccard entre as descrições
num_courses = len(df_des)
jaccard_matrix = [[0]*num_courses for _ in range(num_courses)]
for i in range(num_courses):
    for j in range(i+1, num_courses):
        set1 = set(tfidf_matrix[i].nonzero()[1])
        set2 = set(tfidf_matrix[j].nonzero()[1])
        jaccard_dist = jaccard_similarity(set1, set2)
        jaccard_matrix[i][j] = jaccard_dist
        jaccard_matrix[j][i] = jaccard_dist

# Construir o grafo usando similaridade de cosseno
G_cosine = nx.Graph()
limiar_similaridade = 0.5
for i in range(num_courses):
    for j in range(i+1, num_courses):
        if similarity_matrix_cosine[i][j] > limiar_similaridade:
            G_cosine.add_edge(df_des.loc[i, 'Instituição | Curso'], df_des.loc[j, 'Instituição | Curso'], weight=similarity_matrix_cosine[i][j])

# Construir o grafo usando distância de Jaccard
G_jaccard = nx.Graph()
limiar_distancia = 0.2
for i in range(num_courses):
    for j in range(i+1, num_courses):
        if jaccard_matrix[i][j] < limiar_distancia:
            G_jaccard.add_edge(df_des.loc[i, 'Instituição | Curso'], df_des.loc[j, 'Instituição | Curso'], weight=jaccard_matrix[i][j])

# Salvar os grafos em arquivos
num_nodes_cosine = G_cosine.number_of_nodes()
print("Número de nós (Similaridade de Cosseno):", num_nodes_cosine)
num_edges_cosine = G_cosine.number_of_edges()
print("Número de arestas (Similaridade de Cosseno):", num_edges_cosine)
nx.write_gexf(G_cosine, "cursos_similares_des_cosseno.gexf")

num_nodes_jaccard = G_jaccard.number_of_nodes()
print("Número de nós (Distância de Jaccard):", num_nodes_jaccard)
num_edges_jaccard = G_jaccard.number_of_edges()
print("Número de arestas (Distância de Jaccard):", num_edges_jaccard)
nx.write_gexf(G_jaccard, "cursos_similares_des_jaccard.gexf")
