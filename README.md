# Car-Price-Prediction

**MADE BY:**  
- Diksha Sharma  
- Nupur Gandhi

## Overview

Predicting the market value of used cars can benefit both buyers and sellers. Many individuals are interested in the used car market to either sell their cars or buy used ones. In this project, we aim to predict the price of used cars using various features such as Present_Price, Selling_Price, Kms_Driven, Fuel_Type, and Year. We have employed four regression models: linear regression, random forest regression, gradient booster regression, and XGBoost regressor after cleaning the dataset. Upon fitting the dataset into these models, we found that the XGBoost model is the best for car price prediction, as it generates the least error and provides the best graph. 

## Models Used

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression
4. XGBoost Regressor

## Working of the Code

All the code is available in the .ipynb file. To run the code, open the .ipynb file on Jupyter Notebook or Google Colab. Before running the code, install the required packages:
- Numpy: `pip install numpy`
- Matplotlib: `pip install matplotlib`
- Seaborn: `pip install seaborn`
- Pandas: `pip install pandas`
- Sklearn: `pip install sklearn`

Ensure that both the dataset and the code file are in the same folder or provide the full pathname of the dataset in the code before running it to avoid errors.

## Procedure

In this project, we first prepared our dataset and analyzed it to identify anomalies, making it unsuitable to directly use in regression models. We explored and cleaned the data using specific functions based on our requirements.

### 1) Data Preprocessing

Our first step was Data Preprocessing, analyzing the dataset's present structure, variables, and their descriptions. We imported necessary packages and modules, loaded the dataset, and performed the following steps:
- Exploring Descriptive Statistics
- Dropping unnecessary features
- Checking and treating missing values

### 2) Data Exploration

Our next step involved analyzing how each variable in the dataset is presented and identifying anomalies. The following steps were followed:
- Exploring the Probability Distribution Function (PDF)
- Dealing with outliers
- Checking linearity using scatter plots
- Transforming independent variables using a log-transformation
- Checking multicollinearity using VIF

### 3) Feature Selection

Feature selection techniques were applied to find the best set of features. The following steps were taken:
- Using LabelEncoder()
- Generating a heatmap
- Determining feature importance using ExtraRegressor model
- Converting categorical features using get_dummies function

### 4) Model Development

After selecting important features, we developed the dataset for our model:
- Declaring dependent and independent variables
- Performing feature scaling
- Splitting the dataset into training and test sets

### 5) Linear Regression

We used the linear regression model, fitted it with the training dataset, predicted values, and evaluated its performance using metrics such as r2 score, mean squared error, and mean absolute error.

### 6) Random Forest Regression

The random forest regression model was employed similarly, with fitting, predicting, and evaluating using metrics.

### 7) Gradient Boosting Regression

Gradient boosting regression was implemented with fitting, predicting, and evaluating using metrics.

### 8) XGBoost

XGBoost, an efficient and accurate gradient boosting implementation, was used. Fitting, predicting, and evaluating using metrics were performed.

### 9) Best Model

After working on all three models, we determined the most suitable model by plotting scatter plots of actual vs. predicted prices. XGBoost emerged as the most suitable model for our dataset, producing the least error and the most accurate predicted prices.

### 10) Manually Checking Predictions

For manual verification of predictions, we followed steps to find actual prices, took exponentials of the price column, found residuals, and plotted them in tabular form.

## Conclusion

In conclusion, based on our dataset, the XGBoost Regressor model is the most suitable for predicting car prices.
