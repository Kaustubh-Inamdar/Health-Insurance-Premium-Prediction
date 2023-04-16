# Medical Health Insurance Premium Prediction App

This is an Applied Data Science project that uses machine learning to predict medical health insurance premiums based on the user's health information such as their age, known diseases and allergies, and other features. The project provides two models for prediction: Linear Regression and Ridge Regression. The accuracy of the models is around 65%.

## Table of Contents
* [Project Overview](#project-overview)
* [Tools and Libraries Used](tools-and-libraries-used)
* [How to use the app](#how-to-use-the-app)
* [Data](#data)
* [Modeling](#modeling)
* [Evaluation](#evaluation)
* [Result](#result)
* [Conclusion](#conclusion)

## Project Overview
The goal of this project is to coduct an Exploratory Data Analysis on the data preprocess it and build a machine learning model that predicts the medical health insurance premium of an individual based on their health information. This project is based on a dataset of medical insurance records that contains information such as age, known diseases etc. Creating a webapp using the trained model with Streamlit Library of python.
The project consists of the following steps:

1. Exploratory Data Analysis (EDA)
2. Data Visualization
3. Data Preprocessing
4. Model Training and Testing
5. Creating Webapp using Streamlit that takes user input and predicts the Premium

## Tools and Libraries Used
The project was implemented in Python using the following tools and libraries:

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Streamlit
To install these libraries, use the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit
```

## How to use the app
1. Clone the repository
2. in terminal run: streamlit run predict.py

## Data
The dataset used in this project can be found on Kaggle at: https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction. It contains:
* age: age of primary beneficiary 
* Diabetes: Does individual have diabetes or not
* BloodPressure: Does individual have high blood pressure or not 
* AnyTransplant: Does individual have any transplants
* AnyChronicDiseases: Does individual have any chronic diseases
* Height: Height of an individual in cm
* Weight: Weight of an individual in kg
* KnownAllergies: Known Allergies to individual
* HistoryOfCancerInFamily: Is there any history of cancer in the family
* NumberOfSurgeries: Number of surgeries beneficiary has gone through
* PremiumPrice: individual medical costs billed by health insurance

## Modeling
Two models were used in this project to predict the medical health insurance premium:
* Linear Regression
* Ridge Regression

The models were trained and evaluated using the preprocessed dataset. The Linear Regression model provided an accuracy of 63% while the Ridge Regression model provided an accuracy of 65%.

## Evaluation
The models were evaluated based on their accuracy, which was measured using the r2 score metric. The Ridge Regression model performed slightly better than the Linear Regression model.

## Result
![App1](https://user-images.githubusercontent.com/88809987/232298496-d44de009-d228-4e58-ab34-a0fc8f18e339.png)
![App2](https://user-images.githubusercontent.com/88809987/232298549-d4f00cda-d0d7-4e6e-b4b9-078f04d3944b.png)

## Conclusion
This project shows that it is possible to predict medical health insurance premiums with reasonable accuracy using machine learning models. However, more data and features could be included to further improve the model's performance. Additionally, other models could be explored to see if they provide better accuracy than the ones used in this project.
