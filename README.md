***Rain Prediction in Australia***
This project predicts whether it will rain tomorrow in Australia using the weatherAUS dataset. We use both Logistic Regression and Random Forest classifiers, with a strong focus on handling class imbalance and feature selection to improve performance.

***Dataset***
Source: Bureau of Meteorology, Australia

File: weatherAUS.csv

Target Variable: RainTomorrow (Yes/No)

***Tools & Technologies***
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (Logistic Regression, Random Forest, SelectKBest, SMOTE)

imbalanced-learn (for oversampling)

Jupyter Notebook

***Machine Learning Pipeline***
1. Data Preprocessing
Dropped columns with excessive missing values

Imputed missing values:

Numerical: median

Categorical: mode

Encoded categorical variables using pd.get_dummies

2. Feature Engineering
Correlation and mutual information analysis

Top 20 features selected using SelectKBest + chi2

3. Handling Class Imbalance
Applied SMOTE (Synthetic Minority Over-sampling Technique) on training data

Balanced both classes (RainTomorrow = Yes/No) before training

4. Model Training
Trained two models:

Logistic Regression

Random Forest Classifier

5. Model Evaluation
Used Accuracy, Precision, Recall, and F1-Score

Evaluated class-specific performance, especially for minority class (RainTomorrow = Yes)

***Results***
Model	Accuracy	Rain Recall	Rain F1	Note
Logistic Regression + SMOTE	77.7%	0.77	0.79	Simpler, interpretable model
Random Forest + SMOTE	84.9%	0.85	0.85	Better overall performance

***Key Learnings***
SMOTE significantly improves recall for imbalanced binary classification.

Logistic Regression benefits from feature selection and resampling.

Random Forest handles high-dimensional, categorical-rich data very well.

Adjusting the classification threshold can improve real-world usefulness.

***Future Improvements***
Deploy the model using Streamlit for real-time weather prediction.

Tune hyperparameters using GridSearchCV or RandomizedSearchCV.

Add ROC-AUC and Precision-Recall curves for threshold analysis.

