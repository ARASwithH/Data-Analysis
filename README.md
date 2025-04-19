# 🛍️📈 Store Sales Prediction & 💬 Sentiment Analysis

## 📚 Overview

This repository includes **two distinct projects**:

- **Part I: Store Sales Prediction** – A machine learning project focused on forecasting store sales based on various business and temporal features.
- **Part II: Sentiment Analysis** – A text classification task involving preprocessing, vectorization, and sentiment prediction of customer reviews using multiple ML and NLP techniques.

---

## 📦 Part I: Store Sales Prediction

### 🧾 Dataset

The dataset includes store-level sales data with features like:
- Store ID, Retail Type, Stock Variety
- Promotional campaign info (BOGO)
- Temporal data: Date, Day of Week, Holiday
- Customer count (removed during preprocessing)
- Distance to rival store, rival opening timeline

### 📋 Task Workflow

1. **Load & Merge Datasets**
   - Combine sales and store metadata by `store id`.
2. **Preprocessing**
   - Handle missing values in `DistanceToRivalStore`
   - Extract `Year`, `Month`, `Day`, and `WeekOfYear` from the `Date` column
   - Remove `Customers` and `Date`
   - Apply `StandardScaler` to normalize features
3. **Train/Test Split**
   - Use the first 70% of data (chronologically) for training, remaining 30% for testing
4. **Model Training**
   - Train **Linear Regression** and **Random Forest Regressor**
   - Evaluate their performance
5. **Feature Importance**
   - Use the Random Forest model to identify which features contribute most to predictions

---

## 💬 Part II: Sentiment Analysis

### 🧾 Dataset

A dataset of labeled customer reviews used to train sentiment classification models.

### 📋 Task Workflow

1. **Preprocessing**
   - Convert text to lowercase, remove stopwords and punctuation
2. **Vectorization**
   - **TF-IDF** using `TfidfVectorizer`
   - **Word2Vec** using `gensim.models.Word2Vec`
   - **BERT** embeddings using `SentenceTransformer`
3. **Model Training & Tuning**
   - Trained the following classifiers:
     - Logistic Regression
     - Random Forest
     - K-Nearest Neighbors (KNN)
