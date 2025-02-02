import pandas as pd
import numpy as np
import string
import nltk
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.model_selection  import GridSearchCV , train_test_split
from langdetect import detect 
# preprocessing


# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def clean_text(text):
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def is_english(sentence):  
    try:  
        if detect(sentence) == 'en' :
            return sentence
        else:
            return None
    except:  
        return None  

df = pd.read_csv('train_sentiment.csv')
df = df.drop('Unnamed: 0', axis=1)
df['review'] = df['review'].str.lower()
df = df.drop_duplicates()
df['review'] = df['review'].apply(remove_links)
df['review'] = df['review'].apply(clean_text)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['review'] = df['review'].apply(is_english)
df = df.dropna()
df['rating'] = df['rating'].apply(lambda x: 1 if x>3 else 0)


# tf_idf vectorizer:
vectorizer=TfidfVectorizer(max_features=5000)
x_tfidf=vectorizer.fit_transform(df['review'])

# word2vec vectorizer:
tokenized_reviews = [review.split() for review in df['review']]
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)

def word_2_vec(review):
    words = review.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0) # mean of vectors of the words
    else:
        return np.zeros(word2vec_model.vector_size) # if there was no words in the model, returns zero vecror

x_word2vec = df['review'].apply(word_2_vec) #adding the word2vec vectors to dataset
word2vec_df=pd.DataFrame(x_word2vec)

# bert vectorizer

bert_model = SentenceTransformer("all-MiniLM-L6-v2") 
x_bert=bert_model.encode(df["review"].tolist(), convert_to_numpy=True)
bert_df=pd.DataFrame(x_bert)

# model: regression:
X_train, X_test, y_train, y_test = train_test_split(x_tfidf, df["rating"], test_size=0.2, random_state=42)

param_grid = {'n_estimators': [50, 100, 200]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
f1=f1_score(y_test , y_pred)
print(f1)
