import json
from flask import Flask, render_template, request, jsonify

from Model import Model

app = Flask(__name__)
model = Model()

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/analyze', methods=["POST"])
def Analyze():
    input_data = json.loads(request.values["inp"])
    print("input preprocessing start")
    model.Preprocess(input_data["input-title"], input_data["input-content"])
    print("input preprocessing end")
    print("prediction start")
    model.MakePrediction()
    print("prediction end")
    
    return "ok"


if __name__ == '__main__':
    app.run(debug=True)
