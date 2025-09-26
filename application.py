# Required Libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# Data
data = pickle.load(open("models/df.pkl", "rb"))

# Model
model = pickle.load(open("models/model_rid.pkl", "rb"))

# Scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
        # Getting Data in the form of Dict
        form_data = request.form.to_dict()

        # Seprating Numerical Columns and Categorical Columns
        for key, value in form_data.items():  
            try:
                form_data[key] = float(value)  
            except ValueError:
                pass  
        
        # Creating Dataset of Seperating Columns
        df = pd.DataFrame([form_data])

        # Scaling the Numerical Columns
        Scaling = scaler.transform(df[df.columns])

        # Prediction
        Prediction = model.predict(Scaling)

        # Returning the Result to webpage
        return render_template('home.html',result=Prediction[0])
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")