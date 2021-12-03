import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Model:
    def __init__(self, dataset, lr=0.01):
        # self.save_dir = "model/" # Local
        self.save_dir = "drive/MyDrive/sasw/model/" # Colab
        self.dataset = dataset
        self.batch_size = 64
        self.optimizer = Adam(lr=lr)
        
        best_model_file = "{}{}".format(self.save_dir, "fn_best.h5")
        if os.path.isfile(best_model_file):
            self.model = load_model(best_model_file)
            print("Model loaded from: {}".format(best_model_file))
        else:
            print("Model spawned")
            # Seq Model build        
            self.model = Sequential()
            # embeddidng layer
            self.model.add(Embedding(len(self.dataset.list_of_words), output_dim = 128))
            # Bi-Directional RNN and LSTM
            self.model.add(Bidirectional(LSTM(128)))
            # Dense layers
            self.model.add(Dense(128, activation = 'relu'))
            self.model.add(Dense(1,activation= 'sigmoid'))
            self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['acc'])
            self.model.summary()
    
    def Train(self, epochs=10):
        self.train_score = {"acc":[], "val_acc":[], "loss":[], "val_loss":[]}
        self.best_acc = 0.0
        for ep in range(epochs):
            self.model.fit(self.dataset.train_sequences, np.asarray(self.dataset.y_train), batch_size=self.batch_size, validation_split=0.2, epochs = 1)
            if self.model.history.history["val_acc"][0] > self.best_acc:
                self.best_acc = self.model.history.history["val_acc"][0]
                self.model.save("{}fn_{}.h5".format(self.save_dir, ep + 1))
                print("Model saved with best val_acc {}".format(self.best_acc))
            
            for key in self.model.history.history:
                self.train_score[key].append(self.model.history .history[key][0])
    
    def Predict(self):
        # TODO 
        pass
    
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
        
    def PlotConfMatrix(self):
        # TODO
        pass   
    