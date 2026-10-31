import joblib
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


def evaluate_models(trained_models, X_test, y_test):
    results = []
    for name,model in trained_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision =precision_score(y_test,y_pred,zero_division = 0)
        recall = recall_score(y_test,y_pred,zero_division = 0)
        f1 = f1_score(y_test,y_pred,zero_division = 0) 
        results.append({"Model":name,
                        "Accuracy":accuracy,
                        "Precision":precision,
                        "Recall":recall,
                        "F1 Score":f1})
    results_df= pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No evaluation results found.")
    results_df = results_df.sort_values(by = "F1 Score",ascending = False)
    print("\n Model Evaluation Results:")
    print(results_df.to_string(index =False))
    return results_df
   
def save_best_model(trained_models,results_df):
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model,"churn_model.pkl")
    print("\nBest Model:",best_model_name)
    print("Best modl saved successfully!")
    return best_model
