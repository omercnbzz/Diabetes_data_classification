Overview
This repository contains two files related to diabetes prediction analysis:
diabetes.ipynb - A Jupyter notebook performing diabetes prediction using machine learning models
diabetes.csv - The dataset used for the analysis
Notebook Details: diabetes.ipynb
Data Processing
The dataset is loaded from diabetes.csv using pandas
First 5 rows are displayed to inspect the data structure
The target variable is "Outcome" (1 for diabetic, 0 for non-diabetic)

Data Visualization
A bar chart shows the distribution of diabetes cases:

X-axis: "Healthy (0)" vs "Sick (1)"

Y-axis: Count of cases

Blue bars represent both categories

Grid lines on y-axis for better readability

Title: "Diabetes Status Distribution (Outcome)"

Data Preparation
Train-test split (80% train, 20% test) with random_state=42 for reproducibility

Feature scaling using StandardScaler:

Fit on training data only

Transform both training and test sets

Machine Learning Models
Five classification models are implemented:

Logistic Regression

Accuracy: 75.32%

Precision/Recall:

Class 0: 81%/80%

Class 1: 65%/67%

Confusion matrix shows performance on both classes

K-Nearest Neighbors (KNN)

n_neighbors=5

Accuracy: 69.48%

Precision/Recall:

Class 0: 75%/80%

Class 1: 58%/51%

Decision Tree

random_state=42

(Accuracy and metrics not shown in preview)

Naive Bayes

GaussianNB implementation

(Accuracy and metrics not shown in preview)

Multi-layer Perceptron (MLP)

hidden_layer_sizes=(10,)

max_iter=1000

random_state=42

(Accuracy and metrics not shown in preview)

Evaluation
For each model:

A pipeline is created with StandardScaler and the classifier

Model is trained on scaled training data

Predictions are made on test data

Accuracy score and classification report are printed

Confusion matrix is displayed with:

'Blues' color map

Title indicating the model name

Tight layout for clean visualization

Dataset Details: diabetes.csv
The dataset contains medical predictor variables and a target outcome variable:

Columns:

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age (years)

Outcome: Class variable (0 or 1)

Key Observations
The dataset shows a class imbalance with more non-diabetic cases

Feature scaling is crucial as the variables have different scales

Logistic Regression performed best among the shown models

All models show room for improvement, especially in predicting diabetic cases

How to Use
Clone the repository

Ensure you have Python and Jupyter installed

Install required packages: pandas, scikit-learn, matplotlib, numpy

Run the notebook cells sequentially

Dependencies
Python 3.8

pandas

scikit-learn

matplotlib

numpy

Future Improvements
Address class imbalance using SMOTE or other techniques

Perform more extensive feature engineering

Try ensemble methods like Random Forest or Gradient Boosting

Optimize hyperparameters using GridSearchCV

Add cross-validation for more reliable performance estimates
