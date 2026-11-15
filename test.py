import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("Train_data.csv")

# Load saved model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Apply saved Label Encoders (unknown values become -1)
for col, le in encoders.items():
    known = set(le.classes_)
    df[col] = df[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in known else -1
    )

# Separate features and target (anomaly = 1, normal = 0)
X = df.drop("class", axis=1)
y = df["class"].map({"normal": 0, "anomaly": 1})

# Same split as train.py, so we test on data the model has not seen
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Make predictions (scaling is done inside the saved model)
pred = pd.Series(model.predict(X_test)).map({"normal": 0, "anomaly": 1})

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("\nCYBER ATTACK MODEL TESTING")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))

# Assertions
assert len(pred) == len(X_test)
assert all(p in [0, 1] for p in pred)
assert accuracy > 0.90, "Accuracy is too low!"

print("\nAll test cases passed!")