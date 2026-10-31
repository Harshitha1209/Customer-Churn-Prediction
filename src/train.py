from src.datapreprocessing import load_data, preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression":LogisticRegression(max_iter = 1000),
        "Decision Tree":DecisionTreeClassifier(random_state = 42),
        "Random Forest":RandomForestClassifier(n_estimators = 200,random_state = 42),
        "KNN":KNeighborsClassifier(n_neighbors=5),
        "SVM":SVC(probability=True,random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(random_state = 42)}
    trained_models = {}
    for name,model in models.items():
        model.fit(X_train,y_train)
        trained_models[name] = model
        print(f"{name}training completed")
    return trained_models 

    


