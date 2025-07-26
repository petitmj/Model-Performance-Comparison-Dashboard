# Model Performance Comparison Dashboard  

Model Performance Comparison Dashboard seamlessly integrates **AWS S3 storage, data preprocessing, and multiple ML models** to provide model performance comparisonons.  

## Features  
✅ **Upload and preprocess datasets** (CSV/XLSX)  
✅ **Automated S3 integration** (checks and uploads datasets if missing)  
✅ **Handles missing values** using imputation techniques  


✅ **Supports multiple ML models**, including:  
🤖 Logistic Regression – Best for binary/multiclass classification, medical risk prediction, and simple tabular data  
🌲 Random Forest – Great for tabular data, feature-rich datasets, and handling missing values  
📈 HistGradientBoostingClassifier – Ideal for large tabular datasets with missing values  
⚡ XGBoost – Optimized for large, complex tabular datasets and competitions  
💡 LightGBM – Fast gradient boosting for large tabular datasets  
🐱 CatBoost – Handles categorical features natively, good for mixed-type data  
👥 K-Nearest Neighbors – Works well for small datasets and pattern recognition  
🧠 MLP Neural Network – Suitable for complex relationships and larger datasets  
✅ **Compare performance of selected models side-by-side**
✅ **Model performance metrics** (Accuracy, F1 Score, ROC AUC)  

## How It Works  

1️⃣ Upload a dataset via the **Streamlit interface**  
2️⃣ Select the **target column** and one or more **ML models** to compare  
3️⃣ Train the selected models and view **side-by-side performance metrics**  
4️⃣ Use results to drive **data-driven decision-making** in elder care  

## Technologies Used  

🛠 **Python, Pandas, Scikit-Learn, XGBoost, LightGBM, CatBoost, Boto3, Streamlit**  

---  
