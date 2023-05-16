import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Read the dataset
df = pd.read_csv("/home/chaitanya/Churnp-prediction/Customer_Churn.csv")

# Data exploration
df.head()
df.info()
df["Churn"].value_counts()

# Data visualization
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
numerical = cols

plt.figure(figsize=(20, 4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i + 1)
    sns.countplot(x=str(col), data=df)
    ax.set_title(f"{col}")

sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

cols = ['InternetService', 'TechSupport', 'OnlineBackup', 'Contract']

plt.figure(figsize=(14, 4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i + 1)
    sns.countplot(x="Churn", hue=str(col), data=df)
    ax.set_title(f"{col}")

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

cat_features = df.drop(['customerID', 'TotalCharges', 'MonthlyCharges', 'SeniorCitizen', 'tenure'], axis=1)

le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)

num_features = df[['TotalCharges', 'MonthlyCharges', 'tenure']]
finaldf = pd.concat([num_features, df_cat], axis=1)

# Splitting into training and test sets
X = finaldf.drop("Churn", axis=1)
y = finaldf["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest classifier
rf = RandomForestClassifier(random_state=46)
rf.fit(X_train, y_train)

# Predicting with the trained model
preds = rf.predict(X_test)
print(accuracy_score(preds, y_test))

# Saving the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)
