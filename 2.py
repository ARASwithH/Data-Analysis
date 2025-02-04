import pandas as pd
import numpy as np
import string
import nltk
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.model_selection  import GridSearchCV , train_test_split
from langdetect import detect 
<<<<<<< HEAD
# preprocessing
=======
>>>>>>> 44e560bfe44d7e5bb660c8cc0755d917a8b2e3f2

# Preproseccing

<<<<<<< HEAD
=======
print('Preproseccing...')
>>>>>>> 44e560bfe44d7e5bb660c8cc0755d917a8b2e3f2

print('Downloading NLTK Packages')

nltk.download('stopwords')
nltk.download('punkt')
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

<<<<<<< HEAD
=======
# Data Cleaning
print('Data Cleaning')

>>>>>>> 44e560bfe44d7e5bb660c8cc0755d917a8b2e3f2
df = pd.read_csv('train_sentiment.csv')
df = df.drop('Unnamed: 0', axis=1)
df['review'] = df['review'].str.lower()
df = df.drop_duplicates()
df['review'] = df['review'].apply(remove_links)
<<<<<<< HEAD

=======
>>>>>>> 44e560bfe44d7e5bb660c8cc0755d917a8b2e3f2
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['review'] = df['review'].apply(is_english)
df = df.dropna()
df['rating'] = df['rating'].apply(lambda x: 1 if x>3 else 0)

<<<<<<< HEAD
X_train, X_test, y_train, y_test = train_test_split(df['review'], df["rating"], test_size=0.2, random_state=42)

# tf_idf vectorizer:
vectorizer=TfidfVectorizer(max_features=5000)
x_train_tfidf=vectorizer.fit_transform(X_train)
x_test_tfidf=vectorizer.transform(X_test)

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

X_train_w2v = np.array([word_2_vec(text) for text in X_train])
X_test_w2v = np.array([word_2_vec(text) for text in X_test])


# bert vectorizer

bert_model = SentenceTransformer("all-MiniLM-L6-v2") 
X_train_bert = bert_model.encode(X_train.tolist(), convert_to_numpy=True)
X_test_bert = bert_model.encode(X_test.tolist(), convert_to_numpy=True)



# model: regression:


param_grid = {'n_neighbors': [3, 5, 7]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1', cv=5)
grid.fit(x_train_tfidf, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test_tfidf)
f1=f1_score(y_test , y_pred)
print(f1)

=======
X_train, X_test, y_train, y_test = train_test_split(df['review'], df["rating"], test_size=0.1, random_state=42)

X_train_list ,X_test_list = [], []
names = ['TF-IDF', 'Word2Vec', 'BERT']

print('DONE\n')


# Vetorizition 
print('Vectorizition...')

# TF-IDF Vectorizer:
print('TF-IDF')
vectorizer=TfidfVectorizer(max_features=5000)
x_train_tfidf=vectorizer.fit_transform(X_train)
x_test_tfidf=vectorizer.transform(X_test)
X_train_list.append(x_train_tfidf)
X_test_list.append(x_test_tfidf)

# Word2Vec Vectorizer:
print('Word2Vec')
tokenized_reviews = [review.split() for review in df['review']]
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)

def word_2_vec(review):
    words = review.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0) # mean of vectors of the words
    else:
        return np.zeros(word2vec_model.vector_size) # if there was no words in the model, returns zero vecror

X_train_w2v = np.array([word_2_vec(text) for text in X_train])
X_test_w2v = np.array([word_2_vec(text) for text in X_test])
X_train_list.append(X_train_w2v)
X_test_list.append(X_test_w2v)

# Bert Vectorizer
print('BERT')
bert_model = SentenceTransformer("all-MiniLM-L6-v2") 
X_train_bert = bert_model.encode(X_train.tolist(), convert_to_numpy=True)
X_test_bert = bert_model.encode(X_test.tolist(), convert_to_numpy=True)
X_train_list.append(X_train_bert)
X_test_list.append(X_test_bert)

print('DONE\n')


# Training and Testing
print('Train and Test:')

# KNN Search Grid
print('KNN')
for i in range(3):
    print(names[i],':')
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1', cv=5)
    grid.fit(X_train_list[i], y_train)
    best_model = grid.best_estimator_
    print(best_model)

    y_pred = best_model.predict(X_test_list[i])
    f1=f1_score(y_test , y_pred)
    print('the score of prediction of this model is :',f1)
    print()

# RandomForest Search Grid
print('RandomForest:')

for i in range(3):
    print(names[i],':')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
    }

    grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=5)
    grid.fit(X_train_list[i], y_train)
    best_model = grid.best_estimator_
    print(best_model)

    y_pred = best_model.predict(X_test_list[i])
    f1=f1_score(y_test , y_pred)
    print('the score of prediction of this model is :',f1)
    print()

# LogesticRegression
print('LogeesticRegression:')

for i in range(3):
    print(names[i],':')
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
    }

    grid = GridSearchCV(LogisticRegression(max_iter=100000), param_grid, scoring='f1', cv=5)
    grid.fit(X_train_list[i], y_train)
    best_model = grid.best_estimator_
    print(best_model)

    y_pred = best_model.predict(X_test_list[i])
    f1=f1_score(y_test , y_pred)
    print('the score of prediction of this model is :',f1)
    print()

print('DONE\n')
>>>>>>> 44e560bfe44d7e5bb660c8cc0755d917a8b2e3f2
