from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
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

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
        form_data = request.form.to_dict()
        for key, value in form_data.items():  
            try:
                form_data[key] = float(value)  # Convert if it's a valid number
            except ValueError:
                pass  
        
        df = pd.DataFrame([form_data])
        print(df)
        
        S_scaled = scaler.transform(df[df.columns])

        print('Scaling_Value',S_scaled)
        yp = model.predict(S_scaled)


        return render_template('home.html',result=yp[0])
    

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")