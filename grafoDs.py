import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


#Grafo construido baseao na DESCRiÇÃO do curso

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
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Construço do Grafo
G = nx.Graph()
num_courses = len(df_des)

for curso in df_des['Instituição | Curso']:
    G.add_node(curso)
    
#Define o limiar de similiarade para determinar as arestas que conectam os vétices
limiar_similaridade = 0.5

for i in range(num_courses):
    for j in range(i+1, num_courses):
        if similarity_matrix[i][j] > limiar_similaridade:
            G.add_edge(df_des.loc[i, 'Instituição | Curso'], df_des.loc[j, 'Instituição | Curso'], weight=similarity_matrix[i][j])

# Calcular a similaridade de cosseno entre as descrições e adicionar arestas conforme necessário
for i in range(num_courses):
    for j in range(i+1, num_courses):
        if similarity_matrix[i][j] > limiar_similaridade:
            curso_i = df_des.loc[i, 'Instituição | Curso']
            curso_j = df_des.loc[j, 'Instituição | Curso']
            G.add_edge(curso_i, curso_j, weight=similarity_matrix[i][j])

# Número de nós
num_nodes = G.number_of_nodes()
print("Número de nós:", num_nodes)

# Número de arestas
num_edges = G.number_of_edges()
print("Número de arestas:", num_edges)

# Salvar o grafo em um arquivo
nx.write_gexf(G, "cursos_similares_desc.gexf")

#Exibe a lista de adjacência do curso
lista_adjacencia_curso_i = G.neighbors('Coursera Project Network - Business Strategy: Business Model Canvas Analysis with Miro')

for vizinho in lista_adjacencia_curso_i:
    print(vizinho)
    