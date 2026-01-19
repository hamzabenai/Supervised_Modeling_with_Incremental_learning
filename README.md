---
title: Incremental Learning Platform
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Incremental Learning Platform

An interactive platform for training, retraining, and deploying machine learning models with incremental learning capabilities using River ML.

## Features

- 🎓 **Train New Models**: Upload data and train models from scratch
- 🔄 **Retrain Models**: Update existing models with new data without forgetting
- 🎯 **Make Predictions**: Generate predictions on new datasets
- 📊 **Active Model Management**: Keep track of your current model
- 📈 **Training History**: View past training sessions

## Supported Models

- Random Forest Regressor
- Random Forest Classifier
- Linear Regression
- Logistic Regression

## How to Use

1. **Train a New Model**: Upload your CSV/XLSX data, select target column, and train
2. **Retrain with New Data**: Use the active model or upload one, then add new training data
3. **Make Predictions**: Upload data for predictions using your trained model

## Technology Stack

- **Streamlit**: Interactive web interface
- **River ML**: Online/incremental machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scikit-learn**: Additional ML utilities
