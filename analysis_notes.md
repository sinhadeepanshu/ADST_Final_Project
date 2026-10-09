# Machine Failure Prediction - Complete Data Science Project

## 1. Problem
Predict `Machine failure` from machine operating/sensor variables.

## 2. Dataset
Rows: 10000
Columns: 14
Failure rate: 3.39%

## 3. Important modeling decision
Do not use `UDI`, `Product ID`, or `TWF/HDF/PWF/OSF/RNF` as model inputs.
The first two are identifiers; the failure-mode flags can leak information about the target.

## 4. Workflow
EDA -> preprocessing -> train/test split -> Logistic Regression + Random Forest -> metrics -> save best model -> Streamlit deployment.

## 5. Limitations
This dataset is a static tabular dataset. It does not provide timestamps, machine lifecycle sequences, or a direct remaining-life target. Therefore, a genuine RUL model and genuine 24/48/72-hour forecasting cannot be claimed from this dataset alone. Those are future-scope items, consistent with the project report.

## 6. Re-run code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/ai4i2020.csv")
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df["Machine failure"]

cat = ["Type"]
num = [c for c in features if c not in cat]

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", pre),
    ("model", RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=42, min_samples_leaf=2, n_jobs=-1
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))
```
