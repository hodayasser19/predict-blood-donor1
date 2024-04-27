from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Create flask app
app = Flask(__name__, template_folder='Templates')

# Load joblib model
model = joblib.load("model1.joblib")
scaler = joblib.load('scaler1.joblib')

# Global variable to store the last result
last_result = None


@app.route('/')
def home():
    return render_template('bloodForm.html')


@app.route('/result', methods=["POST"])
def result():
    global last_result

    # Convert input data to numerical values
    hemo = float(request.form['hemo'])
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    haml = int(request.form['haml'])
    smoke = int(request.form['smoke'])
    k7owl = float(request.form['k7owl'])
    aghad = int(request.form['aghad'])
    fshl = int(request.form['fshl'])
    goda = int(request.form['8oda'])

    # Convert data to numpy array
    data = np.array([hemo,age, sex, haml, smoke,k7owl, aghad, fshl, goda]).reshape(1, -1)

    # Scale the data using the trained scaler
    vect = scaler.transform(data)

    # Make prediction
    model_prediction = model.predict(vect)

    # Store the result
    last_result = model_prediction

    return render_template('bloodForm.html', label=model_prediction)


@app.route('/api/result', methods=["GET"])
def get_result():
    global last_result
    if last_result is None:
        return 'No result available'
    else:
        return str(int(last_result[0]))


if __name__ == '__main__':
    app.run()

    