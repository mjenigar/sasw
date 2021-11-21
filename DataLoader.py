import os
import pandas as pd

from tqdm import tqdm

# Obtain additional stopwords from nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

class DataLoader():
    def __init__(self, root, datasets):
        self.root = root
        self.datasets = datasets
        self.dataset = {"title" : [], "text" : [], "label" : [], "clean_words": [], "clean": []}

        self.LOAD = 100
        
    def JoinDatasets(self):
        for dataset in self.datasets:
            for file in os.listdir("{}{}".format(self.root, dataset)):
                if file[0] != "." and file != "submit.csv" and file != "test.csv":
                    full_path = "{}{}/{}".format(self.root, dataset, file)
                    print("Loading from: {}".format(full_path))
                    
                    data = pd.read_csv(full_path)
                    counter = 0
                    for index, row in data.iterrows():
                        
                        if dataset == "data1":
                            label = row.label
                        elif dataset == "data2":
                            label = 1 if file == "True.csv" else 0
                        elif dataset == "data3":
                            label = 1 if row.label == "REAL" else 0
                        
                        self.dataset["title"].append(row.title)
                        self.dataset["text"].append(row.text)
                        self.dataset["label"].append(label)
                        self.dataset["clean_words"].append(None)
                        self.dataset["clean"].append(None)

                        counter += 1
                        
                        if counter > self.LOAD:
                            break
                    print("Loaded: {} rows\n".format(counter))
                
                if counter > self.LOAD:
                        break 
            if counter > self.LOAD:
                break
        print("Total loaded {} rows\n".format(len(self.dataset)))

        self.dataset = pd.DataFrame(self.dataset)
        
    def Preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
                result.append(token)
                
        return result
    
    def CleanData(self):
        for i in tqdm(range(len(self.dataset))):
            self.dataset["clean_words"][i] = self.Preprocess("{} {}".format(self.dataset["title"][i], self.dataset["text"][i]))
            self.dataset["clean"][i] = " ".join(self.dataset["clean_words"][i])
                

    def GetListOfWords(self):
        list_of_words = []
        for sample_words in self.dataset["clean_words"]:
            for word in sample_words:
                list_of_words.append(word)
                
        return list(set(list_of_words))

    
# ROOT = "data/"
# datasets = ["data1", "data2", "data3"]
# d = DataLoader(ROOT, datasets)
# d.JoinDatasets()

# print(d.dataset.head())