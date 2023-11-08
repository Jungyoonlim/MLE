import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
import sklearn.feature_extraction.text
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import numpy as np 
import torch 

'''
Sentiment Analysis on Movie Reviews
This can classify movie reviews into pos or neg categories
- Setting up a data pipeline 
- Performing exploratory data analysis
- Feature engineering
- Training a model
- Evaluating its performance
'''

# steps to follow

#1. dataset -- publicily available movie reviews 

def load(data):
    text = pd.load_csv(data)
    return text

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove special char and numbers
    text = re.sub(r'[\d\W]+','',text)
    # tokenize
    tokens = text.split()
    # remove stop words
    stop_words = set()

    # stemming 
    

    # return preprocessed text

