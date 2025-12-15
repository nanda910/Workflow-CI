import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

data = pd.read_csv("namadataset_preprocessing/StudentsPerformance_preprocessed.csv")

data["average_score"] = data[["math score", "reading score", "writing score"]].mean(axis=1)
threshold = data["average_score"].mean()
data["performance"] = data["average_score"].apply(lambda x: 1 if x >= threshold else 0)

X = data.drop(columns=["performance", "average_score"])
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=y
)

mlflow.set_experiment("Eksperimen_SML_Modelling_Basic")

with mlflow.start_run():

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("precision_manual", prec)
    mlflow.log_metric("recall_manual", rec)
    mlflow.log_metric("f1_score_manual", f1)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
