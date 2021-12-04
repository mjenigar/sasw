import json
from flask import Flask, render_template, request, jsonify
from Models import *

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/analyze', methods=["POST"])
def Analyze():
    report = {
        "random_forest_classifier": None,
        "naive_bayes": None,
        "rnn": None
    }
    
    raw_input = json.loads(request.values["input"])
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

    # try:
    #     model = RNN(None)
    #     pred = model.MakePrediction(x)
    #     report["rnn"] = "1" if pred[0] >= 0.5 else "0"
    # except:
    #     report["rnn"] = "-1"
    
    return json.dumps(report, indent = 2)


if __name__ == '__main__':
    app.run(debug=True)
