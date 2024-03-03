import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

df_skl = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False)
df_skl = df_skl[['Course Name', 'University', 'Skills']]
colunas_plt_ordem = ['University', 'Course Name', 'Skills']
colunas_plt_nome = {'University':'Instituição', 'Course Name':'Título', 'Skills':'Skills'}
df_skl = df_skl.reindex(columns=colunas_plt_ordem)
df_skl = df_skl.rename(columns=colunas_plt_nome)
df_skl['Instituição | Curso'] = df_skl['Instituição'].astype(str) + ' - ' + df_skl['Título'].astype(str)
df_skl = df_skl.dropna()
# Extrair as descrições dos cursos
descriptions = df_skl['Skills']

# Calcular TF-IDF das descrições
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Calcular a similaridade de cosseno entre as descrições
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Construir o grafo
G = nx.Graph()
num_courses = len(df_skl)
for i in range(num_courses):
    limiar_similaridade = 0.5
    for j in range(i+1, num_courses):
        if similarity_matrix[i][j] > limiar_similaridade:  # Defina um limiar de similaridade
            G.add_edge(df_skl.loc[i, 'Instituição | Curso'], df_skl.loc[j, 'Instituição | Curso'], weight=similarity_matrix[i][j])

# Salvar o grafo em um arquivo
# Número de nós
num_nodes = G.number_of_nodes()
print("Número de nós:", num_nodes)

# Número de arestas
num_edges = G.number_of_edges()
print("Número de arestas:", num_edges)

nx.write_gexf(G, "cursos_similares_skills.gexf")