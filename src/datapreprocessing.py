
import pandas as pd
from sklearn.preprocessing import StandardScaler



def load_data():
    df = pd.read_csv("data/Customer_Churn.csv")
    return df
def preprocess(df):
    df.drop(["customerID"],axis = 1,inplace = True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = "coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
    cat_col = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
    df = pd.get_dummies(df,columns = cat_col,drop_first = True,dtype = int)
    
    df = df.dropna()
    
    X = df.drop("Churn",axis = 1)
    y = df["Churn"]

    numerical_cols = ["tenure","MonthlyCharges","TotalCharges"]
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y
