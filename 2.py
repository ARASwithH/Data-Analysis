import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


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

tf_idf=TfidfVectorizer()
tfidf_vectorized=tf_idf.fit_transform(df['review'])

print(tfidf_vectorized)
print(df)





