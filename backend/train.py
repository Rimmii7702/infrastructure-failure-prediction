# src/train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("data/telemetry.csv")

X = df.drop("failure",axis=1)
y = df["failure"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = XGBClassifier()

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Accuracy:",accuracy)

joblib.dump(model,"models/failure_model.pkl")

import matplotlib.pyplot as plt

importance = model.feature_importances_

features = X.columns

plt.barh(features,importance)
plt.tight_layout()
plt.savefig("models/feature_importance.png")