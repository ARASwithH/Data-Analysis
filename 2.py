import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np

# preprocessing


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
df.drop_duplicates()
df['review'] = df['review'].apply(remove_links)
df['review'] = df['review'].apply(clean_text)

# model 2

model_2 = Word2Vec(sentences=df['review'], vector_size=100, window=5, min_count=1, workers=4)
model_2.save("word2vec.model")

def get_sentence_vector(words, model):
    vectors = [model_2.wv[word] for word in words if word in model_2.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model_2.vector_size)

df['vector_2'] = df['review'].apply(lambda words: get_sentence_vector(words, model_2))









