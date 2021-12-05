import os
import re
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

# Obtain additional stopwords from nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, root, datasets, train_size, load=None):
        self.root = root
        self.datasets = datasets
        self.dataset = {"content": [],"clean_words": [], "clean": [], "label" : []}
        
        self.train_size = train_size
        self.test_size = 1 - self.train_size
        
        self.load = load
        
        self.stemmer_eng = SnowballStemmer('english')
        self.words_eng = stopwords.words('english')
        
        print("Loading datasets...")
        self.JoinDatasets()
        print('Cleaning dataset...')
        self.CleanData()
        
        # self.GetListOfWords()
        # self.SplitData(0.7, 0.3)
        # print("DATASET: train {} test {}".format(0.7, 0.3))
        # self.list_of_words = self.GetListOfWords()
        # self.Tokenize()
        
    def JoinDatasets(self):
        counter = 0
        
        for dataset in self.datasets:
            for file in os.listdir("{}{}".format(self.root, dataset)):
                if self.load is not None and counter >= self.load:
                        break
                if file[0] != "." and file != "submit.csv" and file != "test.csv":
                    full_path = "{}{}/{}".format(self.root, dataset, file)
                    print("Loading from: {}".format(full_path))

                    if dataset == "data4":
                        data = pd.read_csv(full_path, encoding='cp1252')
                    else:
                        data = pd.read_csv(full_path)
                        
                    for index, row in data.iterrows():
                        if self.load is not None and counter >= self.load:
                            break
                        if dataset == "data1":
                            label = row.label
                        elif dataset == "data2":
                            label = 1 if file == "True.csv" else 0
                        elif dataset == "data3":
                            label = 1 if row.label == "REAL" else 0
                        elif dataset == "data4": 
                            label = 1 if row.label == "TRUE" else 0
                        
                        
                        if dataset == "data4":
                            if not pd.isna(row.title):
                                content = "{} {} {}".format(row.title, row.text, row.source)
                            else:
                                content = "{} {}".format(row.text, row.source)
                        else:
                            content = "{} {}".format(row.title, row.text)
                        
                        self.dataset["content"].append(content)
                        self.dataset["label"].append(label)
                        self.dataset["clean_words"].append(None)
                        self.dataset["clean"].append(None)
                        counter += 1
                    print("Loaded: {} rows\n".format(counter))
            if self.load is not None and counter >= self.load:
                break       
        print("Total loaded {} rows\n".format(len(self.dataset)))

        self.dataset = pd.DataFrame(self.dataset)
    
    def CleanData(self):
        for i in tqdm(range(len(self.dataset))):
            sample = re.sub(r'http\S+', '', self.dataset["content"][i])
            sample = re.compile(r'<[^>]+>').sub("", sample)
            sample = re.sub('[!|()<>;?$=:+/*,-]|[0-9]|&nbsp', '', sample)
            sample = sample.replace(".", "")
            sample = " ".join(sample.split())

            tokens = word_tokenize(sample)
            tokenWords = [w for w in tokens if w.isalpha()]
            afterStopwords = [w for w in tokenWords if w not in self.words_eng]
            stemmedWords = [self.stemmer_eng.stem(w) for w in afterStopwords]
            output = " ".join(stemmedWords)
            self.dataset["clean"][i] = output

    def GetSample(self, index):
        return self.dataset["clean"][index]
    
    def PlotClassRation(self):
        plt.figure(figsize = (8, 8))
        sns.countplot(y = "label", data = self.dataset)
    
    def SplitData(self, train_size, test_size):
        if (train_size + test_size) != 1.0:
            return False
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset.clean, self.dataset.label, test_size = test_size, shuffle=True)


