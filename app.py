from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')
app = Flask(__name__, static_folder='static')




# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except EOFError:
    print("Error: Unable to load the model from file. Please ensure that the file exists and is not empty.")


# Create a Flask app


# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    Partner = str(request.form[' Partner'])
    Dependents = str(request.form['Dependents'])
    tenure= int(request.form['tenure'])
    PhoneService = str(request.form[ 'PhoneService' ])
    MultipleLines= str(request.form['MultipleLines'])
    InternetService= str(request.form['InternetService'])
    OnlineSecurity= str(request.form['OnlineSecurity'])
    OnlineBackup= str(request.form['OnlineBackup'])
    DeviceProtection= str(request.form['DeviceProtection'])
    TechSupport= str(request.form['TechSupport'])
    StreamingTV= str(request.form[' StreamingTV'])
    StreamingMovies= str(request.form[' StreamingMovies'])
    Contract= str(request.form[' Contract'])
    PaperlessBilling= str(request.form[' PaperlessBilling'])
    PaymentMethod= str(request.form[' PaymentMethod'])
    MonthlyCharges= float(request.form[' MonthlyCharges'])
    TotalCharges= float(request.form[' TotalCharges'])

    # Create a numpy array with the user input values
    input_data = np.array([[age, systolic_bp, diastolic_bp, glucose, body_temp, heart_rate]])
    
    # Use the trained model to make a prediction
    prediction = model.predict(input_data)
    
    # Return the predicted risk level to the user
    return render_template('result.html', prediction_text='The predicted risk level is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)