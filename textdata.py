import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
from nltk.corpus import stopwords
import re

'''
Simple Text Classifier
It categorize text data into two classes
Use a smaller subset of a dataset to keep the processing fast!
'''

# load data
def load_data(filepath):
    dataset = pd.read_csv(filepath)
    return dataset

# preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\d\W]+','', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# feature engineering 


# model training 


# evaluation


# save model

# execution flow 
def main():
