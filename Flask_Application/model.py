import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv("data/airplane-accident-severity.csv")

categorical = ["Violations", "Accident_Type_Code"]
df[categorical] = df[categorical].astype("object")

X = df.drop(columns=["Severity", "Accident_ID"])
y = df["Severity"]

X = pd.get_dummies(X, columns=categorical)
X = X.to_numpy()

label = LabelEncoder()
y = label.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=201
)

classifier = GradientBoostingClassifier(
    n_estimators=100, max_depth=10, learning_rate=0.2, random_state=101
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

pickle.dump(classifier, open("gb_classifier.pkl", "wb"))
