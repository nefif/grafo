from sklearn.feature_extraction.text import TfidfVectorizer

# Exemplo de lista de documentos
documents = [
    "By the end of this guided project, you will be fluent in identifying and mapping forces for Force Field Analysis using a hands-on example. This will enable you to map and rate the forces which is important in for validating, preparing, and managing change in professional and personal life.   Change happens all the time and in being able to identify factors involved in change and preparing to manage change you increase your chances for success. This analysis will help you if you are in: + Strategy development + Program Management + Project Management + Business Process Re-Engineering + Product Development + Organisational Development And much more. On a personal level this analysis can help you to map Forces for Change and Forces Against Change for different settings. For example: + Competing in sports + Having a professional goal + Developing a good habit  Furthermore, this guided project is designed to engage and harness your visionary and exploratory abilities. And further equip you with the knowledge to utilise the learned concepts, methodologies, and tools to prepare for change in various settings."
]

# Inicialize o vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Ajuste o vetorizador e transforme os documentos em uma matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Obtenha os nomes dos termos (palavras-chave)
terms = tfidf_vectorizer.get_feature_names_out()

# Para cada documento, obtenha as palavras-chave com os maiores valores de TF-IDF
for i, doc in enumerate(documents):
    print(f"Documento {i+1}:")
    # Obtenha os índices dos termos com os maiores valores de TF-IDF
    top_indices = tfidf_matrix[i].toarray().argsort()[0][::-1][:10]
    # Imprima as palavras-chave
    keywords = [terms[idx] for idx in top_indices]
    print("Palavras-chave:", keywords)
    print()