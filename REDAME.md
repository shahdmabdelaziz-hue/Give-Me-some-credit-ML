Credit Default Prediction

A Machine Learning project that predicts whether a borrower may experience serious financial delinquency within the next two years.

This project is based on the Give Me Some Credit Kaggle competition.

🔗 "Kaggle Competition" (https://www.kaggle.com/competitions/GiveMeSomeCredit/leaderboard)

📌 Project Overview

The goal of this project is to build and compare different Machine Learning and Deep Learning approaches for credit-risk prediction and anomaly detection.

The target variable is:

- "SeriousDlqin2yrs"
  - "0" → No serious delinquency
  - "1" → Serious delinquency within two years

The dataset contains 125,113 records and 11 features after removing the unnecessary index column.

📊 Dataset

The main features include:

- "RevolvingUtilizationOfUnsecuredLines"
- "age"
- "NumberOfTime30-59DaysPastDueNotWorse"
- "DebtRatio"
- "MonthlyIncome"
- "NumberOfOpenCreditLinesAndLoans"
- "NumberOfTimes90DaysLate"
- "NumberRealEstateLoansOrLines"
- "NumberOfTime60-89DaysPastDueNotWorse"
- "NumberOfDependents"

Missing Values

Missing values were found mainly in:

Feature| Missing Values
MonthlyIncome| 24,803
NumberOfDependents| 3,272
NumberOfOpenCreditLinesAndLoans| 1
NumberOfTimes90DaysLate| 1
NumberRealEstateLoansOrLines| 1
NumberOfTime60-89DaysPastDueNotWorse| 1

🔧 Data Preprocessing

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Removed the unnecessary "Unnamed: 0" column.
3. Checked missing values.
4. Handled missing values.
5. Explored the numerical features using statistical analysis and box plots.
6. Treated outliers.
7. Split the data into training and testing sets using:
   - "test_size = 0.2"
   - "random_state = 42"
8. Applied feature scaling where required.

Train/Test Split

- Training set: 100,090 samples
- Test set: 25,023 samples
- Number of features: 10

🤖 Models

Several models were implemented and evaluated:

1. Logistic Regression

Used as a baseline classification model.

Test Accuracy: 93.90%

Class| Precision| Recall| F1-score
0| 0.94| 0.99| 0.97
1| 0.64| 0.17| 0.27

Test Accuracy: "0.9390"

---

2. Decision Tree

Used to capture nonlinear relationships between the features.

Test Accuracy: 89.90%

Class| Precision| Recall| F1-score
0| 0.95| 0.94| 0.95
1| 0.26| 0.29| 0.27

Test Accuracy: "0.8990"

---

3. Random Forest

An ensemble model based on multiple decision trees.

Test Accuracy: 93.73%

Class| Precision| Recall| F1-score
0| 0.95| 0.99| 0.97
1| 0.57| 0.20| 0.30

Test Accuracy: "0.9373"

---

4. XGBoost

A gradient boosting model for binary classification.

Parameters included:

- "n_estimators = 200"
- "learning_rate = 0.1"
- "random_state = 42"

Test Accuracy: 93.77%

Class| Precision| Recall| F1-score
0| 0.95| 0.99| 0.97
1| 0.58| 0.20| 0.30

Test Accuracy: "0.9377"

---

5. Isolation Forest

Isolation Forest was used as an unsupervised anomaly detection approach.

The model was configured with:

- "contamination = 0.01"
- "random_state = 42"

It detected:

- 1,001 outliers
- 1.00% of the training data

When compared against the original target:

Training Data

Class| Precision| Recall| F1-score
0| 0.94| 0.99| 0.97
1| 0.53| 0.08| 0.14

Test Data

Class| Precision| Recall| F1-score
0| 0.94| 1.00| 0.97
1| 0.61| 0.09| 0.16

---

6. Weighted XGBoost

To address the class imbalance, a weighted XGBoost model was trained using "scale_pos_weight".

Test Data

Class| Precision| Recall| F1-score
0| 0.98| 0.83| 0.90
1| 0.23| 0.74| 0.35

Test Accuracy: 82%

The weighted model significantly increased the recall for the positive class:

Recall = 74%

compared with 20% for the standard XGBoost model.

---

7. Autoencoder

A neural-network-based Autoencoder was implemented for anomaly detection.

Architecture:

Input
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Output

Training configuration:

- Epochs: 50
- Batch size: 256
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Threshold: 85th percentile of reconstruction error

Test Data

Class| Precision| Recall| F1-score
0| 0.96| 0.88| 0.92
1| 0.24| 0.54| 0.33

Test Accuracy: 86%

📈 Model Results Summary

Model| Test Accuracy| Positive-Class Precision| Positive-Class Recall| Positive-Class F1
Logistic Regression| 93.90%| 0.64| 0.17| 0.27
Decision Tree| 89.90%| 0.26| 0.29| 0.27
Random Forest| 93.73%| 0.57| 0.20| 0.30
XGBoost| 93.77%| 0.58| 0.20| 0.30
Weighted XGBoost| 82.00%| 0.23| 0.74| 0.35
Autoencoder| 86.00%| 0.24| 0.54| 0.33

«Note: The dataset is highly imbalanced, so Accuracy alone does not fully describe model performance. Precision, Recall, and F1-score for the positive class are also reported.»

🔍 Key Findings

- The standard classification models achieved around 90–94% test accuracy.
- Logistic Regression achieved 93.90% test accuracy.
- Random Forest achieved 93.73% test accuracy.
- Standard XGBoost achieved 93.77% test accuracy.
- Weighted XGBoost increased positive-class recall to 74%, compared with 20% for standard XGBoost.
- The Autoencoder achieved 54% recall for the positive class.
- The Isolation Forest detected 1% of the training data as anomalies based on the selected contamination value.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Google Colab
- Kaggle

📂 Project Structure

Credit-Default-Prediction/
│
├── project_credit_2.ipynb
├── README.md
└── dataset/
    └── cs-training.csv

«Dataset files are not included in this repository. They can be downloaded from Kaggle.»

🚀 How to Run

1. Clone the repository.
2. Download the dataset from Kaggle.
3. Open "project_credit_2.ipynb" using Google Colab or Jupyter Notebook.
4. Upload "cs-training.csv".
5. Run the notebook cells sequentially.

🎯 Project Objectives

This project provides practical experience in:

- Data preprocessing
- Missing-value handling
- Outlier detection and treatment
- Data visualization
- Binary classification
- Imbalanced datasets
- Ensemble learning
- Anomaly detection
- Deep learning
- Model evaluation

📚 Reference

Kaggle – Give Me Some Credit

https://www.kaggle.com/competitions/GiveMeSomeCredit
