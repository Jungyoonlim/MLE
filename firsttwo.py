# task 1: loads a dataset from a CSV and inspect to understand its structure and format
import pandas as pd
import os 

def read_csv(filepath):
    if os.path.isfile(filepath):
        try: 
            df = pd.read_csv(filepath)
            # Loading the CSV into a data structure that allows for easy data manipulation and inspection
            # Display a small part of the dataset to get an overview of the data
            print(df.head())
            # data types
            print(df.dtypes)
            # statistical summary of numeric cols
            print(df.describe())
            return df 

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("File not found")
    
    return df 

# task 2: Custom Preprocessing Function 
'''
- convert the text to lowercase
- replace non-letter characters with spaces
- tokenize the text into individual words
- remove english stopwords 
- apply stemming to the words
- combine the words back into a single string

'''
from nltk.corpus import stopwords 
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = text.lower()

    text = re.sub(r'')

    tokens = text.split()

    stemmed = 

    return ' '.join(stemmed)
