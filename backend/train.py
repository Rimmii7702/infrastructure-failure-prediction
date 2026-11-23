from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

BACKEND_DIR = Path(__file__).resolve().parent
DATA_PATH = BACKEND_DIR.parent / "data" / "telemetry.csv"
MODELS_DIR = BACKEND_DIR / "models"

df = pd.read_csv(DATA_PATH)

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

joblib.dump(model, MODELS_DIR / "failure_model.pkl")

import matplotlib.pyplot as plt

importance = model.feature_importances_

features = X.columns

plt.barh(features,importance)
plt.tight_layout()
plt.savefig(MODELS_DIR / "feature_importance.png")
