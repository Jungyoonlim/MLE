# task 1: loads a dataset from a CSV and inspect to understand its structure and format
import pandas as pd
import os 

def read_csv(filepath):
    if os.path.isfile(filepath):
        try: 
            text = pd.read_csv(filepath)

            
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("File not found")
    
    return text 




