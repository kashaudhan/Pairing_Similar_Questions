from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from generate_features import feature_create

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def templet():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    questions = [x for x in request.form.values()]
    features = feature_create(questions[0], questions[1])
    prob = model.predict(features)
    if prob > 0.4 :
        return render_template('index.html', pred = 'Similar questions')
    else:
        return render_template('index.html', pred = 'Not similar questions')

if __name__ == "__main__":
    app.run(debug = True)
