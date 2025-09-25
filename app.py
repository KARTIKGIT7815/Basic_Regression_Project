from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## Data
data = pickle.load(open("models/df.pkl", "rb"))

## Model
model = pickle.load(open("models/model_rid.pkl", "rb"))

## Scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# column_names = data.columns
# columns_df = pd.DataFrame(column_names, columns=["Column Name"])

# columns_df.to_csv("column_names.csv", index=False)

@app.route("/")
def hello_world():
    return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")