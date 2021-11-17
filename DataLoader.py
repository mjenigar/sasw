import os
import pandas as pd

class DataLoader():
    def __init__(self, root, datasets):
        self.root = root
        self.datasets = datasets
        self.dataset = []

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
                        
                        self.dataset.append({ "title": row.title, "text": row.text, "label": label })
                        counter += 1
                    print("Loaded: {} rows\n".format(counter))
        print("Total loaded {} rows\n".format(len(self.dataset)))

ROOT = "data/"
datasets = ["data1", "data2", "data3"]
d = DataLoader(ROOT, datasets)
d.JoinDatasets()


print(d.dataset[42])