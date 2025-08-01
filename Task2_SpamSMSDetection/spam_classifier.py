# spam_classifier.py (No Comments)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

DATA_PATH = 'data/spam.csv'
PLOTS_DIR = 'plots'
MODEL_DIR = '.'
MODEL_NAME = 'spam_classifier_model.pkl'
VECTORIZER_NAME = 'tfidf_vectorizer.pkl'

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_style('whitegrid')

def load_and_preprocess_data(path):
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(path, encoding='latin1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except FileNotFoundError:
        print(f"Error: Dataset not found at {path}")
        return None
    
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Data loaded and preprocessed successfully.")
    return df

def clean_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    cleaned = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(cleaned)

def perform_eda(df):
    print("Performing Exploratory Data Analysis...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df.assign(label=df['label'].map({0: 'Ham', 1: 'Spam'})))
    plt.title('Distribution of Spam vs. Ham')
    plt.savefig(os.path.join(PLOTS_DIR, 'spam_distribution.png'))
    plt.close()

    df['message_length'] = df['message'].apply(len)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='message_length', hue=df['label'].map({0: 'Ham', 1: 'Spam'}), bins=50, multiple='stack')
    plt.title('Message Length Distribution by Label')
    plt.savefig(os.path.join(PLOTS_DIR, 'message_length_distribution.png'))
    plt.close()

    ham_words = ' '.join(df[df['label'] == 0]['cleaned_message'])
    spam_words = ' '.join(df[df['label'] == 1]['cleaned_message'])

    ham_wc = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(ham_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Ham Messages')
    plt.savefig(os.path.join(PLOTS_DIR, 'wordcloud_ham.png'))
    plt.close()

    spam_wc = WordCloud(width=800, height=400, background_color='black').generate(spam_words)
    plt.figure(figsize=(10, 8))
    plt.imshow(spam_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Spam Messages')
    plt.savefig(os.path.join(PLOTS_DIR, 'wordcloud_spam.png'))
    plt.close()
    
    print("EDA plots saved to 'plots/' directory.")

def train_and_evaluate(X_train, y_train, X_test, y_test):
    print("Training and evaluating models...")
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine": SVC(random_state=42)
    }
    
    best_model = None
    best_accuracy = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        plt.close()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
    print(f"\nBest model selected: {type(best_model).__name__} with Accuracy: {best_accuracy:.4f}")
    return best_model

if __name__ == '__main__':
    df = load_and_preprocess_data(DATA_PATH)
    
    if df is not None:
        print("Cleaning text messages...")
        df['cleaned_message'] = df['message'].apply(clean_text)
        
        perform_eda(df.copy())
        
        print("Performing feature extraction and splitting data...")
        X = df['cleaned_message']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        final_model = train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test)
        
        print("Saving model and vectorizer...")
        joblib.dump(final_model, os.path.join(MODEL_DIR, MODEL_NAME))
        joblib.dump(tfidf_vectorizer, os.path.join(MODEL_DIR, VECTORIZER_NAME))
        print(f"Model saved as '{MODEL_NAME}'")
        print(f"Vectorizer saved as '{VECTORIZER_NAME}'")
        
        print("\n--- Example Prediction ---")
        sample_spam = "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/claim-now to claim now."
        sample_ham = "Hey, are we still on for dinner tonight at 7? Let me know."
        
        cleaned_spam = clean_text(sample_spam)
        cleaned_ham = clean_text(sample_ham)
        
        sample_spam_tfidf = tfidf_vectorizer.transform([cleaned_spam])
        sample_ham_tfidf = tfidf_vectorizer.transform([cleaned_ham])
        
        prediction_spam = final_model.predict(sample_spam_tfidf)
        prediction_ham = final_model.predict(sample_ham_tfidf)
        
        print(f"Message: '{sample_spam}'")
        print(f"Prediction: {'Spam' if prediction_spam[0] == 1 else 'Ham'}")
        
        print(f"\nMessage: '{sample_ham}'")
        print(f"Prediction: {'Spam' if prediction_ham[0] == 1 else 'Ham'}")
        
        print("\n--- Project Execution Complete ---")