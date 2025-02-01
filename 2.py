import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split


#preprocessing

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def clean_text(text):
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df = pd.read_csv('train_sentiment.csv')
df = df.drop('Unnamed: 0', axis=1)
df['review'] = df['review'].str.lower()
df = df.drop_duplicates()
df['review'] = df['review'].apply(remove_links)
df['review'] = df['review'].apply(clean_text)


# spliting train and test

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['rating'], test_size=0.2, random_state=42)


# vectorizing

# TF-IDF
tf_idf = TfidfVectorizer(min_df=2)
X_train_tfidf = tf_idf.fit_transform(X_train)
X_test_tfidf = tf_idf.transform(X_test)

# Word2Vec
sentences = [nltk.word_tokenize(text) for text in X_train]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_sentence_vector(text, model):
    words = nltk.word_tokenize(text)  
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_w2v = np.array([get_sentence_vector(text, w2v) for text in X_train])
X_test_w2v = np.array([get_sentence_vector(text, w2v) for text in X_test])

# BERT
bert = SentenceTransformer("all-MiniLM-L6-v2")
X_train_bert = bert.encode(X_train)
X_test_bert = bert.encode(X_test)
