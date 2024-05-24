import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import graphviz


# Carregar dados do CSV
@st.cache_data
def carregar_dados():
    df_skl = pd.read_csv('Coursera.csv', delimiter=',', low_memory=False)
    df_skl = df_skl[['Course Name', 'University', 'Skills']]
    colunas_plt_ordem = ['University', 'Course Name', 'Skills']
    colunas_plt_nome = {'University': 'Instituição', 'Course Name': 'Título', 'Skills': 'Skills'}
    df_skl = df_skl.reindex(columns=colunas_plt_ordem)
    df_skl = df_skl.rename(columns=colunas_plt_nome)
    df_skl['Instituição | Curso'] = df_skl['Instituição'].astype(str) + ' - ' + df_skl['Título'].astype(str)
    df_skl = df_skl.dropna()
    return df_skl

# Extrair descrições e calcular TF-IDF
@st.cache_data
def calcular_tfidf(descriptions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    return tfidf_matrix

# Calcular similaridade de cosseno
@st.cache_data
def calcular_similaridade(_tfidf_matrix):
    similarity_matrix = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
    return similarity_matrix

# Função para gerar o grafo
def gerar_grafo(df, similarity_matrix, limiar_similaridade):
    G = nx.Graph()
    num_courses = len(df)

    for curso in df['Instituição | Curso']:
        G.add_node(curso)

    for i in range(num_courses):
        for j in range(i+1, num_courses):
            if similarity_matrix[i][j] > limiar_similaridade:
                curso_i = df.loc[i, 'Instituição | Curso']
                curso_j = df.loc[j, 'Instituição | Curso']
                G.add_edge(curso_i, curso_j, weight=similarity_matrix[i][j])
    return G

# Carregar dados
df_skl = carregar_dados()

# Exibir interface do usuário
st.title('Visualização de Trilhas de Aprendizagem')

# Dropdown para selecionar o curso
selected_course = st.selectbox("Selecione um curso:", df_skl['Instituição | Curso'])

# Slider para escolher o limiar de similaridade
limiar_similaridade = st.slider("Escolha o limiar de similaridade:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Extrair descrições e calcular TF-IDF
descriptions = df_skl['Skills']
tfidf_matrix = calcular_tfidf(descriptions)

# Calcular similaridade de cosseno
similarity_matrix = calcular_similaridade(tfidf_matrix)

# Gerar o grafo
G = gerar_grafo(df_skl, similarity_matrix, limiar_similaridade)

# Exibir o grafo
st.graphviz_chart(graphviz.Source(nx.nx_agraph.to_agraph(G).to_string()))

# Exibir nome dos cursos e pesos das arestas
st.subheader("Cursos e Pesos das Arestas:")
for u, v, w in G.edges(data=True):
    st.write(f"Cursos: {u} -> {v}, Peso: {w['weight']}")
