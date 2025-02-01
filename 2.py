import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np

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
df['tokens'] = df['review'].apply(lambda x: x.split())

#model_1
tf_idf = TfidfVectorizer(min_df=2)
tfidf_vectorized = tf_idf.fit_transform(df['review'])
feature_names = tf_idf.get_feature_names_out()
df['vector_1'] = list(tfidf_vectorized.toarray())


#model_2
model_2 = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
model_2.save("word2vec.model")

def get_sentence_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df['vector_2'] = df['tokens'].apply(lambda x: get_sentence_vector(x, model_2))

#model_3
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
df['vector_3'] = list(bert_model.encode(df['review'].tolist()))

print(df)
