import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import string

# Download dos recursos necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Texto de exemplo
texto = "Write a Full Length Feature Film Script  In this course, you will write a complete, feature-length screenplay for film or television, be it a serious drama or romantic comedy or anything in between. You�ll learn to break down the creative process into components, and you�ll discover a structured process that allows you to produce a polished and pitch-ready script by the end of the course. Completing this project will increase your confidence in your ideas and abilities, and you�ll feel prepared to pitch your first script and get started on your next. This is a course designed to tap into your creativity and is based in ""Active Learning"". Most of the actual learning takes place within your own activities - that is, writing! You will learn by doing.  Here is a link to a TRAILER for the course. To view the trailer, please copy and paste the link into your browser. https://vimeo.com/382067900/b78b800dc0  Learner review: ""Love the approach Professor Wheeler takes towards this course. It's to the point, easy to follow, and very informative! Would definitely recommend it to anyone who is interested in taking a Screenplay Writing course!  The course curriculum is simple: We will adopt a professional writers room process in which you�ll write, post your work for peer review, share feedback with your peers and revise your work with the feedback you receive from your peers. That's how we do it in the real world. You will feel as if you were in a professional writers room yet no prior experience as a writer is required. I'm a proponent of Experiential Learning (Active Learning). My lectures are short (sometimes just two minutes long) and to the point, designed in a step-by-step process essential to your success as a script writer. I will guide you but I won�t ""show"" you how to write. I firmly believe that the only way to become a writer is to write, write, write.  Learner Review: ""I would like to thank this course instructor. It's an amazing course""  What you�ll need to get started: As mentioned above, no prior script writing experience is required. To begin with, any basic word processor will do. During week two, you can choose to download some free scriptwriting software such as Celtx or Trelby or you may choose to purchase Final Draft, the industry standard, or you can continue to use your word processor and do your own script formatting.   Learner Review: ""Now I am a writer!""  If you have any concerns regarding the protection of your original work, Coursera's privacy policy protects the learner's IP and you are indeed the sole owners of your work."

# Tokenização do texto
tokens = word_tokenize(texto.lower())

# Remoção de stopwords e pontuações
stop_words = set(stopwords.words('english'))
stop_words.update(set(string.punctuation))  # Adiciona pontuações
filtered_tokens = [word for word in tokens if word not in stop_words]

# Lematização
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Frequência das palavras
fdist = FreqDist(lemmatized_tokens)

# Listar as 10 palavras mais comuns
palavras_chave = fdist.most_common(10)
print(palavras_chave)
