from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

app = Flask(__name__)

# Load the model from a file
with open('model/model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        crime_type = int(request.form['type'])  # Get the type input from the dropdown
        
        # Make prediction including the type input
        prediction = loaded_model.predict(np.array([[year, month, crime_type]]))
        
        return render_template('index.html', prediction=prediction[0])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
