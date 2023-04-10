import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from flask import Flask,request,render_template,jsonify
from flask_cors import CORS

import pickle

#creating an instance of flask
app = Flask(__name__)
CORS(app)


@app.route("/",methods=['POST'])
def home():
    dataset_url="Dataset\winequality-red.csv"
    dataread = pd.read_csv(dataset_url)

    

    #taking the data

    fixed_acidity = request.form.get('fixed acidity')
    volatile_acidity = request.form.get('volatile acidity')
    citric_acid = request.form.get('citric acid')
    residual_sugar = request.form.get('residual sugar')
    chlorides = request.form.get('chlorides')
    free_sulfur_dioxide = request.form.get('free sulfur dioxide')
    total_sulfur_dioxide = request.form.get('total sulfur dioxide')
    density = request.form.get('density')
    pH = request.form.get('pH')
    sulphates = request.form.get('sulphates')
    alcohol = request.form.get('alcohol')

    load_model=pickle.load(open("model.sav","rb"))

    input_datas = [float(fixed_acidity),float(volatile_acidity),float(citric_acid),float(residual_sugar),float(chlorides),float(free_sulfur_dioxide),float(total_sulfur_dioxide),float(density),float(pH),float(sulphates),float(alcohol)]
    input_array =np.array(input_datas)
    input_datas_reshaped = input_array.reshape(1,-1)
    predictions = load_model.predict(input_datas_reshaped)

    print(predictions)

    if (predictions[0] == 1):
        output ="Good Wine Quality"
    else:
        output ="Bad Wine Quality"
        
    response = jsonify({'output': output})
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



# Running app
if __name__ == '__main__':
    app.run(debug=True)
