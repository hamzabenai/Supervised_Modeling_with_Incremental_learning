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

# 🤖 Incremental Learning Platform

An interactive web platform for training, retraining, and deploying machine learning models with incremental learning capabilities powered by River ML.

## 🌟 Features

- **🎓 Train New Models**: Upload your dataset (CSV/XLSX) and train machine learning models from scratch
- **🔄 Retrain Existing Models**: Update your models with new data while preserving previous knowledge (avoiding catastrophic forgetting)
- **🎯 Make Predictions**: Generate predictions on new datasets using your trained models
- **📊 Active Model Management**: Seamlessly track and manage your current active model
- **📈 Training History**: View and manage your past training sessions
- **💾 Model Persistence**: Save and reload models for future use

## 🛠️ Supported Models

### Regression Models
- **Linear Regression**: Simple linear regression for continuous targets
- **Random Forest Regressor**: Ensemble method for robust regression tasks

### Classification Models
- **Logistic Regression**: Binary and multi-class classification
- **Random Forest Classifier**: Ensemble classifier for complex classification problems

## 📖 How to Use

### 1️⃣ Train a New Model
1. Navigate to the **"Train New Model"** tab
2. Upload your training data (CSV or XLSX format)
3. Select your target column (the variable you want to predict)
4. Choose identifier columns to exclude (optional)
5. Select the model type (Regressor or Classifier)
6. Click **"Train Model"**
7. Download your trained model or keep it active for retraining/predictions

### 2️⃣ Retrain an Existing Model
1. Go to the **"Retrain Model"** tab
2. Upload new training data
3. Choose to use your active model or upload a different one
4. Click **"Retrain Model"**
5. Your model will learn from the new data while retaining previous knowledge

### 3️⃣ Make Predictions
1. Navigate to the **"Make Predictions"** tab
2. Upload the data you want predictions for
3. Use your active model or upload a trained model
4. Click **"Generate Predictions"**
5. View results and download predictions as CSV

## 🔧 Technology Stack

- **Streamlit**: Interactive web interface
- **River ML**: Online/incremental machine learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities and preprocessing
- **Altair**: Declarative statistical visualizations

## 🧠 Incremental Learning

This platform uses **incremental learning** (also known as online learning), which allows models to:
- Learn continuously from streaming data
- Update without retraining from scratch
- Avoid catastrophic forgetting using replay buffers
- Handle concept drift in evolving data distributions

## 📊 Data Requirements

### Training Data
- **Format**: CSV or XLSX
- **Structure**: Tabular data with headers
- **Size**: Any size (system automatically adjusts preprocessing based on dataset size)
- **Missing Values**: Handled automatically based on data size and column characteristics

### Prediction Data
- **Format**: CSV or XLSX
- **Columns**: Must match the features used during training (excluding target column)

## 🎯 Use Cases

- **Continuous Learning**: Update models as new data arrives
- **A/B Testing**: Train multiple model versions and compare
- **Real-time Predictions**: Deploy models for instant predictions
- **Educational**: Learn about incremental learning and online ML
- **Prototyping**: Quickly test ML ideas without complex setup

## 📝 Example Workflow

```
1. Upload sales_data_2023.csv → Train RandomForestRegressor
2. Model achieves score of 0.85
3. New data arrives (sales_data_2024_q1.csv)
4. Retrain model with new data
5. Model score improves to 0.88
6. Upload unseen_customers.csv for predictions
7. Download predictions.csv with results
```

## 🔒 Privacy & Data

- All data processing happens in your session
- Models and data are not stored permanently on servers
- Download your models to keep them for future use

## 🤝 Contributing

This is an open-source educational project. Contributions, issues, and feature requests are welcome!

## 📚 Learn More

- [River ML Documentation](https://riverml.xyz/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Incremental Learning Concepts](https://en.wikipedia.org/wiki/Incremental_learning)

## 📄 License

MIT License - feel free to use this project for learning and development!

---

**Built with ❤️ using River ML and Streamlit**
