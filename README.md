# MedicalCostPersonalRegresionAI
This project revolves around predicting individual medical costs billed by health insurance using regression analysis. The primary components of the project are the Flask web application and the associated Jupyter Notebook for data exploration and model training.
### ***Created by Team23***

<br>

## Attribute information:
###### Age: age of primary beneficiary
###### Sex: Male or Female
###### BMI: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
###### Children: Number of children covered by health insurance / Number of dependents
###### Smoker: The beneficiary smokes or does not smoke
###### Region: The beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
###### Charges: Individual medical costs billed by health insurance

### Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance

<br>

## Flask Web Application
The Flask web application serves as the user interface for interacting with the trained regression model. It includes the following key features:
### 1. Data Preprocessing and Visualization
- ###### Data Loading and Preprocessing: The insurance dataset is loaded and preprocessed to prepare it for model training.
- ###### Label Encoding: Categorical features such as 'sex', 'smoker', and 'region' are encoded for compatibility with machine learning models.
- ###### Standard Scaling: Numeric features are standardized to ensure consistency in model training.
- ###### Pairplot Visualization: Seaborn is employed to create a pairplot, providing insights into relationships between features with a focus on the 'smoker' attribute.
- ###### Age vs. Charges Scatter Plot: A scatter plot visualizing the relationship between age and medical charges, with 'smoker' highlighted for additional context.
### 2. Linear Regression Model Training
- ###### Train-Test Split: The dataset is split into training and testing sets to facilitate model evaluation.
- ###### Linear Regression Model: Utilizes scikit-learn to train a linear regression model based on the preprocessed data.
### 3. Model Evaluation
- ###### R-squared Values: Calculates R-squared values for both the training and testing sets to assess model accuracy.
- ###### Mean Squared Errors (MSE): Evaluates the model performance using MSE for both training and testing sets.
### 4. Web Interface for Predictions
- ###### User Interface: A user-friendly interface allows users to input their information for cost predictions.
- ###### Prediction Display: Renders predictions based on the linear regression model, ensuring non-negativity of predictions.

<br>

## Demonstaration:
![Medical Cost Predictor](prints/Medical%20Cost%20Predictor-1.png)

<br>

## Clone the Repository
```bash
git clone https://github.com/BogdanYuskevich/MedicalCostPersonalRegresionAI
cd MedicalCostPersonalRegresionAI
