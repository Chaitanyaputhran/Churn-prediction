# Customer Churn Prediction

This project demonstrates customer churn prediction using a Random Forest Classifier in Python.

## Introduction

Customer churn, or the loss of customers, can have a significant impact on businesses. This code aims to address the problem of customer churn by predicting which customers are likely to leave a service or product. By identifying these customers early, businesses can take proactive measures to retain them. The Random Forest Classifier is used as the predictive model for this task.

## Getting Started

To run this code and use the customer churn prediction model, follow these steps:

1. **Environment Setup:**

   - Make sure you have Python installed on your system.
   - Install the required Python libraries using `pip`:

     ```
     pip install pandas scikit-learn matplotlib seaborn
     ```

2. **Clone the Repository:**

   Clone this GitHub repository to your local machine:
      ```
     git clone https://github.com/Chaitanyaputhran/Churn-prediction.git
     ```


4. **Data Preparation:**

- Place your dataset in the project directory with the filename `Customer_Churn.csv`.
- Run the provided code to explore and preprocess the data.

4. **Model Training:**

- Run the code to train the Random Forest Classifier using the preprocessed data.
- The trained model will be saved as `model.pkl`.

5. **Model Deployment:**

- Use the saved model to make predictions on new data.
- The LabelEncoder used for encoding target variables is saved as `label_encoder.pkl`.

## Data Exploration

The data exploration process involved analyzing the dataset to gain insights into customer behavior. Visualizations were created to better understand the data.



## Data Preprocessing

Data preprocessing steps include handling missing values and encoding categorical variables. These steps are crucial for preparing the data for machine learning.

## Model Training

The Random Forest Classifier was chosen as the predictive model due to its effectiveness in handling complex datasets. The code includes default hyperparameters, but you can fine-tune them for better performance if needed.

## Model Evaluation

The model's performance was evaluated using accuracy as the metric. The results can be found in the code output. Further evaluation metrics can be added as needed.

## Files Included

- `Customer_Churn.csv`: The dataset used for training and testing.
- `model.pkl`: The trained Random Forest Classifier model.
- `label_encoder.pkl`: The LabelEncoder used for encoding target variables.


