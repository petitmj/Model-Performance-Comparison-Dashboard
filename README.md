# AI ElderCare Dashboard  

AI ElderCare Dashboard is a **machine learning-powered analytics platform** designed to assist healthcare professionals and caregivers in **predicting patient outcomes** and **analyzing elderly care data**. The dashboard seamlessly integrates **AWS S3 storage, data preprocessing, and multiple ML models** to provide insightful predictions.  

## Features  
✅ **Upload and preprocess datasets** (CSV/XLSX)  
✅ **Automated S3 integration** (checks and uploads datasets if missing)  
✅ **Handles missing values** using imputation techniques  
✅ **Supports multiple ML models**, including:  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- HistGradientBoostingClassifier (handles NaNs natively)  
- XGBoost (optimized for large datasets)  
✅ **Model performance metrics** (Accuracy, F1 Score, ROC AUC)  

## How It Works  
1️⃣ Upload a dataset via the **Streamlit interface**  
2️⃣ Select the **target column** and an **ML model**  
3️⃣ Train the model and view **performance metrics**  
4️⃣ Use results to drive **data-driven decision-making** in elder care  

## Technologies Used  
🛠 **Python, Pandas, Scikit-Learn, XGBoost, Boto3, Streamlit**  

---  
