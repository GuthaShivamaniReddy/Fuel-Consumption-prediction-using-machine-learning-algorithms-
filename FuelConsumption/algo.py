import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'fuel_consumption_classification.csv'
df = pd.read_csv(file_path)

# Encode categorical features
label_encoders = {}
for column in df.columns[:-2]:  # All columns except the last (label) and fuelconsumption
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split dataset into features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

print(results)

# Input data to predict
input_data = ["Male", "60", "Postgraduate", "Employee", "1 year", "Car", "Leased", "Engine Issue", "Dry", 20]

# Encode the input data using the label encoders
input_encoded = []
for i, column in enumerate(df.columns[:-2]):  # Encode all categorical columns
    input_encoded.append(label_encoders[column].transform([input_data[i]])[0])
input_encoded.append(input_data[-1])  # Append the numeric fuelconsumption value
input_encoded = np.array(input_encoded).reshape(1, -1)

# Predict using Logistic Regression
logistic_model = KNeighborsClassifier(n_neighbors=5)
logistic_model.fit(X_train, y_train)

prediction = logistic_model.predict(input_encoded)
print(prediction)
