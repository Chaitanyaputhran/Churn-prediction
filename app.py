from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    partner = request.form['Partner']
    dependents = request.form['Dependents']
    tenure = int(request.form['tenure'])
    phone_service = request.form['PhoneService']
    multiple_lines = request.form['MultipleLines']
    internet_service = request.form['InternetService']
    online_security = request.form['OnlineSecurity']
    online_backup = request.form['OnlineBackup']
    device_protection = request.form['DeviceProtection']
    tech_support = request.form['TechSupport']
    streaming_tv = request.form['StreamingTV']
    streaming_movies = request.form['StreamingMovies']
    contract = request.form['Contract']
    paperless_billing = request.form['PaperlessBilling']
    payment_method = request.form['PaymentMethod']
    monthly_charges = float(request.form['MonthlyCharges'])
    total_charges = float(request.form['TotalCharges'])

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Preprocess the input data
    cat_features = input_data.drop(['TotalCharges', 'MonthlyCharges', 'tenure'], axis=1)
    cat_features = cat_features.apply(lambda col: le.transform(col.astype(str)))
    num_features = input_data[['TotalCharges', 'MonthlyCharges', 'tenure']]
    input_data_processed = pd.concat([num_features, cat_features], axis=1)

    # Make prediction
    churn_prediction = model.predict(input_data_processed)
    if churn_prediction[0] == 1:
        result = "Customer is likely to churn."
    else:
        result = "Customer is likely to stay."

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
