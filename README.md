# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecommunications company.

Customer churn is a major challenge for businesses because losing existing customers can affect revenue and business growth.

This project uses customer demographic information, subscription details, service usage and billing information to predict customer churn.

A **Streamlit web application** allows users to enter customer details and receive real-time churn predictions along with churn probability.

## 🎯 Project Objectives

- Analyze customer information and identify churn patterns.
- Perform data cleaning and preprocessing.
- Apply different Machine Learning classification algorithms.
- Compare model performance using evaluation metrics.
- Identify the best-performing model.
- Develop an interactive Streamlit application for customer churn prediction.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Data Preprocessing | Pandas, StandardScaler |
| Model Evaluation | Accuracy, Precision, Recall, F1 Score |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── Customer_Churn.csv
│
├── src/
│   ├── __init__.py
│   ├── datapreprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset Description

The project uses a telecommunications customer churn dataset containing customer information such as:

- Customer demographics
- Tenure
- Internet and phone services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

**Target Variable:**
- `0` – Customer stays
- `1` – Customer churns

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

- Loaded the customer churn dataset using Pandas.
- Removed unnecessary customer ID column.
- Converted TotalCharges into numeric format.
- Handled missing values using median imputation.
- Converted the target variable into binary format.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.

### 2. Train-Test Split

Split the dataset into:

- Training data: 80%
- Testing data: 20%

Used stratified sampling to maintain the target class distribution.

### 3. Model Training

Implemented and compared six Machine Learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gradient Boosting

### 4. Model Evaluation

Evaluated model performance using:

- Accuracy
- Precision
- Recall
- F1 Score

### 5. Best Model Selection

Logistic Regression achieved the highest F1 Score among the six models evaluated in the current experiment.

The selected model, scaler and feature-column information are saved using Joblib for application deployment.

## 📈 Model Performance

The following results were obtained during model evaluation:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.41% | 65.22% | 56.15% | 60.34% |
| Gradient Boosting | 79.99% | 65.75% | 51.34% | 57.66% |
| SVM | 79.49% | 64.81% | 49.73% | 56.28% |
| KNN | 76.86% | 56.56% | 55.35% | 55.95% |
| Random Forest | 78.92% | 63.90% | 47.33% | 54.38% |
| Decision Tree | 73.67% | 50.36% | 56.42% | 53.22% |

**Best Performing Model:** Logistic Regression

**F1 Score:** 60.34%

*Note: These are the results from the current train-test experiment. Performance may vary with different data splits and configurations.*

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application where users can:

- Enter customer demographic details.
- Select subscription and service information.
- Enter tenure and billing details.
- Predict whether a customer is likely to churn.
- View the predicted churn probability.

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Customer-Churn-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install  streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

### Step 6: Train the Models

```bash
python main.py
```

This generates the trained model, scaler and feature-column files.

### Step 7: Run the Streamlit Application

```bash
streamlit run app.py
```

## 📌 Key Learnings

- Data cleaning and preprocessing.
- Categorical feature encoding.
- Numerical feature scaling.
- Training multiple classification algorithms.
- Comparing classification performance.
- Model serialization using Joblib.
- Building an interactive ML application using Streamlit.
- Integrating a trained ML model with a web application.

## 🔮 Future Enhancements

- Hyperparameter tuning for improved model performance.
- Handle class imbalance using SMOTE and other techniques.
- Optimize classification threshold to improve churn detection.
- Add confusion matrix and ROC-AUC visualizations.
- Include feature importance and explainable AI techniques.
- Deploy the application to Streamlit Community Cloud.

## 👩‍💻 Author

**Harshitha Vallepu**

B.Tech Graduate | Aspiring Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/YOUR-USERNAME

LinkedIn: Add your LinkedIn profile URL here.

---

⭐ If you find this project useful, feel free to star the repository!
