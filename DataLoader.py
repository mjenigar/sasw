import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, root, datasets):
        self.root = root
        self.datasets = datasets
        self.dataset = {"title" : [], "text" : [], "label" : [], "clean_words": [], "clean": []}
        
        print("Loading datasets...")
        self.JoinDatasets()
        print('Cleaning dataset...')
        self.CleanData()
        self.SplitData(0.7, 0.3)
        print("DATASET: train {} test {}".format(0.7, 0.3))
        
        self.list_of_words = self.GetListOfWords()
        self.Tokenize()
        
        
    def JoinDatasets(self):
        counter = 0
        
        for dataset in self.datasets:
            for file in os.listdir("{}{}".format(self.root, dataset)):
                if counter >= 64:
                        break
                    
                if file[0] != "." and file != "submit.csv" and file != "test.csv":
                    full_path = "{}{}/{}".format(self.root, dataset, file)
                    print("Loading from: {}".format(full_path))

                    data = pd.read_csv(full_path)
                    for index, row in data.iterrows():
                        if counter > 64:
                            break
                        
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
                    print("Loaded: {} rows\n".format(counter))
            if counter >= 64:
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
    
    def SplitData(self, train_size, test_size):
        if (train_size + test_size) != 1.0:
            return False
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset.clean, self.dataset.label, test_size = test_size, shuffle=True)

    def Tokenize(self, max_len=5000):
        # Create a tokenizer to tokenize the words and create sequences of tokenized words
        tokenizer = Tokenizer(num_words = len(self.list_of_words))
        tokenizer.fit_on_texts(self.x_train)
        self.train_sequences = tokenizer.texts_to_sequences(self.x_train)
        self.train_sequences = pad_sequences(self.train_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
        self.test_sequences = tokenizer.texts_to_sequences(self.x_test)
        self.test_sequences = pad_sequences(self.test_sequences, maxlen = max_len, truncating = 'post') 
