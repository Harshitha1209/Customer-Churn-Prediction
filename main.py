import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.datapreprocessing import load_data, preprocess
from src.train import train_model
from src.evaluate import evaluate_models,save_best_model

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state = 42,stratify = y)

    numerical_cols = ["tenure","MonthlyCharges","TotalCharges"]
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    trained_models = train_model(X_train,y_train)

    results_df = evaluate_models(trained_models,X_test,y_test)
    best_model = save_best_model(trained_models,results_df)
     
    joblib.dump(scaler,"scaler.pkl")
    # Save feature columns
    joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

    print("Feature columns saved successfully!")
    print("Scaler saved successfully!")