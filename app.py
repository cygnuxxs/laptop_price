import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from joblib import load
import warnings
from flask_pymongo import PyMongo

from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)

with open('model.pkl', 'rb') as f:
    pipe_loaded = pickle.load(f)

model = load('model.pkl')

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/laptopprices'
mongo = PyMongo(app)

db = mongo.db.prices

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        company = request.form['company']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        cpu_brand = request.form['cpu_brand']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        os = request.form['os']

        # Create a DataFrame from the user input
        laptop = {
            'Company': [company],
            'Ram': [ram],
            'Weight': [weight],
            'Cpu brand': [cpu_brand],
            'HDD': [hdd],
            'SSD': [ssd],
            'os': [os]
        }

        user_input_df = pd.DataFrame(laptop)

        # Make a prediction using the loaded model
        predicted_log_price = model.predict(user_input_df)
        predicted_price = np.exp(predicted_log_price)[0]
        laptop_dict ={
            'Company': company,
            'Ram': ram,
            'Weight': weight,
            'Cpu brand': cpu_brand,
            'HDD': hdd,
            'SSD': ssd,
            'os': os,
            'price' : round(float(predicted_price), 2)
        }
        db.insert_one(laptop_dict)

        # Render the result page with the predicted price
        return render_template('result.html', price=round(float(predicted_price), 2))
    
if __name__ == '__main__':
    app.run(debug=True)

