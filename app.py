from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction= model.predict(features)
    return render_template("index.html", prediction_text = "The predicted value of Soil available Cd is {} (mg/kg)".format(prediction))




if __name__ == "__main__":
    app.run(debug=True)
