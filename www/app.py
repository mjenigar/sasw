import json

from flask import Flask, render_template, request, jsonify

from Database import Database
from Helpers import *
from Models import *

app = Flask(__name__)
database = Database("localhost", "sasw", "mjenigar", "SaswDB123!")

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/articles')
def Articles():
    return render_template("articles.html")

@app.route('/analyze', methods=["POST"])
def Analyze():
    raw_input = json.loads(request.values["input"])
    report = {
        "random_forest_classifier": None,
        "naive_bayes": None,
        "rnn": None
    }
    x = "{} {} {}".format(raw_input["input-title"], raw_input["input-content"], raw_input["input-src"])

    try:
        model = ModelRandomForestClassifier(None)
        model.LoadModel("rfc.pkl")
        report["random_forest_classifier"] = str(model.MakePrediction(x))
    except:
        report["random_forest_classifier"] = "-1"
    
    try:
        model = NaiveBayes(None)
        model.LoadModel("nb.pkl")
        report["naive_bayes"] = str(model.MakePrediction(x))
    except:
        report["naive_bayes"] = "-1"

    try:
        model = RNN(None)
        pred = model.MakePrediction(x)
        report["rnn"] = "1" if pred[0] >= 0.5 else "0"
    except:
        report["rnn"] = "-1"
    
    predictions = [int(report["random_forest_classifier"]), int(report["naive_bayes"]), int(report["rnn"])]
    toDB = Raw2Dict(raw_input, predictions)
    if database.Connect():
        database.InsertRecord(toDB)
        database.Disconnect()

    return json.dumps(report, indent = 2)

@app.route('/get_records', methods=["POST"])
def GetRecords():
    search = request.values["search"]
    if len(search) == 0 or search == None:
        search = None
        
    if database.Connect():
        records = database.GetRecords(search)
        database.Disconnect()
        return json.dumps(records, default=str)
    else:
        return "fail"
    
if __name__ == '__main__':
    app.run(debug=True)
