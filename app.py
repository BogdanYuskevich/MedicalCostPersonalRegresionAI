# app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# Load and preprocess the dataset
dataset = pd.read_csv('data/insurance.csv')
encoder = LabelEncoder()
dataset['sex'] = encoder.fit_transform(dataset['sex'])
dataset['smoker'] = encoder.fit_transform(dataset['smoker'])
dataset['region'] = encoder.fit_transform(dataset['region'])

X = dataset.drop('charges', axis=1)
y = dataset['charges']

# Visualization - Pairplot
sns.pairplot(dataset, hue='smoker', diag_kind='kde')
plt.title('Pairplot of Features with Smoker Highlighted')
plt.savefig('static/pairplot.png')  # Save pairplot image for display in HTML

# Visualization - Age vs. Charges
plt.figure(figsize=(12, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=dataset)
plt.title('Scatter plot of Age vs Charges with Smoker Highlighted')
plt.savefig('static/age_vs_charges.png')  # Save age vs charges scatter plot image for display in HTML

# Train the Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_train = linear_model.predict(X_train_scaled)
y_pred_test = linear_model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"R-squared (Train): {r2_train:.4f}")
print(f"R-squared (Test): {r2_test:.4f}")
print(f"Mean Squared Error (Train): {mse_train:.2f}")
print(f"Mean Squared Error (Test): {mse_test:.2f}")

# Render the HTML form
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])

    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    input_data_scaled = scaler.transform(input_data)

    prediction_linear = max(linear_model.predict(input_data_scaled)[0], 0)  # Ensure prediction is not negative

    return render_template('index.html', prediction=f'Cost: {prediction_linear:.2f}$')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
