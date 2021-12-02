class Score:
    def __init__(self):
        self.train_score = {"acc" : [], "val_acc": [], "loss" : [], "val_loss"}
        self.batch_train_score = {"acc" : [], "val_acc": [], "loss" : [], "val_loss"}
        self.accuracy = [0] * 2
        self.loss = [0] * 2
        
    def AvgTrainScore(self):
        for key in self.train_score:
            self.train_score[key].append(np.average(self.batch_train_score[key]))
        
        self.accuracy[0] = np.average(self.batch_train_score["acc"])
        self.accuracy[1] = np.average(self.batch_train_score["val_acc"])
        self.loss[0] = np.average(self.batch_train_score["loss"])
        self.loss[1] = np.average(self.batch_train_score["val_loss"])
        
        for key in self.batch_train_score:
            self.batch_train_score[key] = []
        
        return self.accuracy, self.loss
    
    def PrintRunTrainScore(self, cur_ep, total_ep):
        print("EP: {}/{} Train Loss: {:.4f} Train Accuracy: {:.4f} Val Loss: {:.4f} Val Accuracy: {:.4f}".format(cur_ep, total_ep, self.loss[0], self.accuracy[0], self.loss[1], self.accuracy[1]))
    
    def PrintTrainScore(self):
        print("\nTrain Loss: {:.4f} Train Accuracy: {:.4f} Val Loss: {:.4f} Val Accuracy: {:.4f}".format(np.average(self.train_score["loss"]), np.average(self.train_score["accuracy"]), np.average(self.train_score["val_loss"]), np.average(self.train_score["val_accuracy"])))    
    
    def PlotTrainScore(self, att, path_to_save=None):
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

        if path_to_save is not None:
            plt.savefig(path_to_save)

        plt.show()