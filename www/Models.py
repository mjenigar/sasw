import os 
import re
import tqdm
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

class RNN:
    def __init__(self, dataset, lr=0.001):
        # self.save_dir = "drive/MyDrive/sasw/model/" # Colab
        self.save_dir = "models/" # Local
        self.dataset = dataset
        self.optimizer = Adam(learning_rate=lr)
        try:
            self.vectorizer = self.LoadModel("vectorizer.pkl")
            print("trained features loaded")
        except:
            self.vectorizer = TfidfVectorizer()
            print("new vectorizer spawned")
        
        best_model_file = "{}{}".format(self.save_dir, "fn.h5")
        if os.path.isfile(best_model_file):
            self.model = load_model(best_model_file)
            print("Model loaded from: {}".format(best_model_file))
        else:
            print("Model spawned")
            self.list_of_words = self.GetListOfWords()
            # Seq Model build        
            self.model = Sequential()
            # embeddidng layer
            self.model.add(Embedding(len(self.list_of_words), output_dim = 128))
            # Bi-Directional RNN and LSTM
            self.model.add(Bidirectional(LSTM(128)))
            # Dense layers
            self.model.add(Dense(128, activation = 'relu'))
            self.model.add(Dense(1,activation= 'sigmoid'))
            self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['acc'])
            self.model.summary()
            
            self.SplitData(0.7, 0.3)
            self.Tokenize()
    
    def SplitData(self, train_size, test_size):
        if (train_size + test_size) != 1.0:
            return False
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset.dataset.clean, self.dataset.dataset.label, test_size = test_size, shuffle=True)
        
    def GetListOfWords(self):
        list_of_words = []
        for sample in self.dataset.dataset["clean"]:
            for word in sample.split():
                list_of_words.append(word)
                
        return list(set(list_of_words))
    
    def Train(self, epochs=10, batch_size=64):
        self.train_score = {"acc":[], "val_acc":[], "loss":[], "val_loss":[]}
        self.best_acc = 0.0
        for ep in range(epochs):
            self.model.fit(self.train_sequences, np.asarray(self.y_train), batch_size=batch_size, validation_split=0.2, epochs = 1)
            if self.model.history.history["val_acc"][0] > self.best_acc:
                self.best_acc = self.model.history.history["val_acc"][0]
                self.model.save("{}fn_{}.h5".format(self.save_dir, ep + 1))
                print("Model saved new best val_acc is: {}".format(self.best_acc))
            
            for key in self.model.history.history:
                self.train_score[key].append(self.model.history .history[key][0])
    
    def Predict(self, save_to):
        self.predictions = self.model.predict(self.test_sequences)
        predicted_results = []
        for i in range(len(self.predictions)):
            if self.predictions[i].item() > 0.5:
                predicted_results.append(1)
            else:
                predicted_results.append(0)
        
        accuracy = accuracy_score(list(self.y_test), predicted_results)
        cm = confusion_matrix(list(self.y_test), predicted_results)
        plt.figure(figsize = (25, 25))
        sns.heatmap(cm, annot = True)
        print("Model Accuracy: {}".format(accuracy))
        if save_to is not None:
            plt.savefig(save_to)
    
    def Tokenize(self, max_len=5000):
        # Create a tokenizer to tokenize the words and create sequences of tokenized words
        tokenizer = Tokenizer(num_words = len(self.list_of_words))
        tokenizer.fit_on_texts(self.x_train)
        self.train_sequences = tokenizer.texts_to_sequences(self.x_train)
        self.train_sequences = pad_sequences(self.train_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
        self.test_sequences = tokenizer.texts_to_sequences(self.x_test)
        self.test_sequences = pad_sequences(self.test_sequences, maxlen = max_len, truncating = 'post') 
    
    def PlotScore(self, att, save_to):
        y_train = np.array(self.train_score[att])
        y_valid = np.array(self.train_score["val_{}".format(att)])
        x = np.arange(1, len(y_train)+1, 1)
        
        plt.figure()
        plt.plot(x, y_valid)
        plt.plot(x, y_train, color='red')
        plt.legend(("Validation {}".format(att),"Training {}".format(att)))
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.title("{} of validation and training set".format(att))
        plt.grid()

        if save_to is not None:
            plt.savefig(save_to)

        plt.show()   

    def PreprocessSample(self, sample):
        stemmer_eng = SnowballStemmer('english')
        words_eng = stopwords.words('english')
        
        sample = re.sub(r'http\S+', '', sample)
        sample = re.compile(r'<[^>]+>').sub("", sample)
        sample = re.sub('[!|()<>;?$=:+/*,-]|[0-9]|&nbsp', '', sample)
        sample = sample.replace(".", "")
        sample = " ".join(sample.split())

        tokens = word_tokenize(sample)
        tokenWords = [w for w in tokens if w.isalpha()]
        afterStopwords = [w for w in tokenWords if w not in words_eng]
        stemmedWords = [stemmer_eng.stem(w) for w in afterStopwords]
        output = " ".join(stemmedWords)
        
        return self.vectorizer.transform([output]).toarray()
    
    def MakePrediction(self, x):
        X = self.PreprocessSample(x)
        prediction = self.model.predict(X)
        
        return prediction[0]

    def LoadModel(self, name):
        try:
            with open("{}{}".format(self.save_dir, name), 'rb') as f:
                self.model = pickle.load(f)

            return self.model
        except:
            print("model file {}{} missing".format(self.save_dir, name))
            return None
    
class ModelRandomForestClassifier:
    def __init__(self, dataset):
        self.save_dir = "models/"
        self.dataset = dataset
        
        try:
            self.vectorizer = self.LoadModel("vectorizer.pkl")
            print("trained features loaded")
        except:
            self.vectorizer = TfidfVectorizer()
            print("new vectorizer spawned")

        self.classifier = RandomForestClassifier()
        
    def PreprocessData(self):
        y = self.dataset.dataset["label"]
        X = self.vectorizer.fit_transform(self.dataset.dataset.clean).toarray()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, self.dataset.dataset.label, test_size = self.dataset.test_size, shuffle=True)

    def PreprocessSample(self, sample):
        stemmer_eng = SnowballStemmer('english')
        words_eng = stopwords.words('english')
        
        sample = re.sub(r'http\S+', '', sample)
        sample = re.compile(r'<[^>]+>').sub("", sample)
        sample = re.sub('[!|()<>;?$=:+/*,-]|[0-9]|&nbsp', '', sample)
        sample = sample.replace(".", "")
        sample = " ".join(sample.split())

        tokens = word_tokenize(sample)
        tokenWords = [w for w in tokens if w.isalpha()]
        afterStopwords = [w for w in tokenWords if w not in words_eng]
        stemmedWords = [stemmer_eng.stem(w) for w in afterStopwords]
        output = " ".join(stemmedWords)
        
        return self.vectorizer.transform([output]).toarray()
    
    def MakePrediction(self, x):
        X = self.PreprocessSample(x)
        prediction = self.model.predict(X)
        
        return prediction[0]
            
    def Train(self):
        self.model = self.classifier.fit(self.x_train, self.y_train)

    def Predict(self):
        self.y_pred = self.model.predict(self.x_test)

        precision, recall, fscore, train_support = score(self.y_test, self.y_pred, pos_label=1, average='binary')
        print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round(accuracy_score(self.y_test, self.y_pred), 3)))

        cm = confusion_matrix(self.y_test, self.y_pred)
        class_label = [0, 1]
        df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
    
    def SaveModel(self):
        with open("{}{}".format(self.save_dir, "rfc.pkl"), 'wb') as fid:
            pickle.dump(self.model, fid)
            
        with open("{}{}".format(self.save_dir, "vectorizer.pkl"), 'wb') as fid:
            pickle.dump(self.vectorizer, fid)

    def LoadModel(self, name):
        try:
            with open("{}{}".format(self.save_dir, name), 'rb') as f:
                self.model = pickle.load(f)

            return self.model
        except:
            print("model file {}{} missing".format(self.save_dir, name))
            return None
        

class NaiveBayes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.save_dir = "models/"
        try:
            self.vectorizer = self.LoadModel("vectorizer_nb.pkl")
            print("trained features loaded")
        except:
            self.vectorizer = CountVectorizer()
            print("new vectorizer spawned")

        self.classifier = MultinomialNB()
        
    def PreprocessData(self):
        y = self.dataset.dataset["label"]
        X = self.vectorizer.fit_transform(self.dataset.dataset.clean).toarray()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, self.dataset.dataset.label, test_size = self.dataset.test_size, shuffle=True) 
        
    def Train(self):
        self.model = self.classifier.fit(self.x_train, self.y_train)
        
    def Predict(self):
        self.y_pred = self.model.predict(self.x_test)

        precision, recall, fscore, train_support = score(self.y_test, self.y_pred, pos_label=1, average='binary')
        print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round(accuracy_score(self.y_test, self.y_pred), 3)))

        cm = confusion_matrix(self.y_test, self.y_pred)
        class_label = [0, 1]
        df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
    
    def PreprocessSample(self, sample):
        stemmer_eng = SnowballStemmer('english')
        words_eng = stopwords.words('english')
        
        sample = re.sub(r'http\S+', '', sample)
        sample = re.compile(r'<[^>]+>').sub("", sample)
        sample = re.sub('[!|()<>;?$=:+/*,-]|[0-9]|&nbsp', '', sample)
        sample = sample.replace(".", "")
        sample = " ".join(sample.split())

        tokens = word_tokenize(sample)
        tokenWords = [w for w in tokens if w.isalpha()]
        afterStopwords = [w for w in tokenWords if w not in words_eng]
        stemmedWords = [stemmer_eng.stem(w) for w in afterStopwords]
        output = " ".join(stemmedWords)
        
        return output
    
    def MakePrediction(self, x):
        sample = self.PreprocessSample(x)
        X = self.vectorizer.transform([sample]).toarray()
        prediction = self.model.predict(X)
        
        return prediction[0]
    
    def SaveModel(self):
        with open("{}{}".format(self.save_dir, "nb.pkl"), 'wb') as fid:
            pickle.dump(self.model, fid)
            
        with open("{}{}".format(self.save_dir, "vectorizer_nb.pkl"), 'wb') as fid:
            pickle.dump(self.vectorizer, fid)

    def LoadModel(self, name):
        try:
            with open("{}{}".format(self.save_dir, name), 'rb') as f:
                self.model = pickle.load(f)

            return self.model
        except:
            print("model file {}{} missing".format(self.save_dir, name))
            return None
    