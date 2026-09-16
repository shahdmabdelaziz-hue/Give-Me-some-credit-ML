Credit Default Prediction

Machine Learning project based on the Give Me Some Credit Kaggle competition.

Project Overview

The goal is to predict whether a borrower will experience serious financial delinquency within two years.

Data Preprocessing

- Loaded and cleaned the dataset using Pandas
- Handled missing values
- Explored statistics and correlations
- Detected and treated outliers
- Split data into 80% training / 20% testing
- Applied feature scaling where needed

Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Isolation Forest
- Weighted XGBoost
- Autoencoder

Results

Model| Accuracy| Recall
Logistic Regression| 93.90%| 17%
Decision Tree| 89.90%| 29%
Random Forest| 93.73%| 20%
XGBoost| 93.77%| 20%
Weighted XGBoost| 82%| 74%
Autoencoder| 86%| 54%

Kaggle Score

0.86274

Technologies

Python • Pandas • NumPy • Scikit-learn • XGBoost • TensorFlow/Keras • Matplotlib • Seaborn

Project File

"project_credit_2.ipynb"
