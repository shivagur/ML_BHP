import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


#This line creates an instance of the Flask class and assigns it to a variable app.
app=Flask(__name__)

#These two lines load the pre-trained machine learning model and the scaler object from the pickle files 'regmodel.pkl' and 'scaling.pkl',  pickle.load() method is used to load the contents of the pickle files.

regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

# When a user visits the home page, the home() function is called. This function renders an HTML template called 'home.html' using the render_template() method.
@app.route('/')
def home():
    return render_template('home.html')

'''This block of code creates a route decorator that maps to the '/predict_api' URL endpoint.
    When a user submits a POST request to this endpoint, the predict_api() function is called. 
    The function retrieves the data sent in the POST request in JSON format using request.json['data']. 
    Then, the function converts the JSON object into a 1D NumPy array using np.array() and then reshapes it into a 2D array using the reshape() method. 
    After that, the function scales the data using the scalar.transform() method, which scales the data using the previously loaded scaler object. 
    Finally, the function predicts the house price using the pre-trained machine learning model 
    and returns the predicted output in JSON format using the jsonify() method.'''

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

'''This block of code creates a route decorator that maps to the '/predict' URL endpoint. 
    When a user submits a POST request to this endpoint, the predict() function is called. 
    The function retrieves the data sent in the POST request as a list of floats using request.form.values(). 
    Then, the function converts this list into a 1D NumPy array using np.array() and then reshapes it into a 2D array using the reshape() method. 
    After that, the function scales the data using the previously loaded scaler object using `scalar.transform'''

@app.route('/predict',methods=['GET','POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     