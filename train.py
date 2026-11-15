import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load data
data = pd.read_csv("Train_data.csv")

# 2. Convert text columns to numbers
text_columns = ["protocol_type", "service", "flag"]
encoders = {}
for col in text_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# 3. Split features and target
X = data.drop("class", axis=1)
y = data["class"]

# 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Three models with a few settings to try (hyperparameter tuning)

models = {
    "Decision Tree": (
        make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42)),
        {"decisiontreeclassifier__max_depth": [5, 10, 20]},
    ),
    "SVM": (
        make_pipeline(StandardScaler(), SVC()),
        {"svc__C": [1, 10], "svc__kernel": ["linear", "rbf"]},
    ),
    "Random Forest": (
        make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
        {"randomforestclassifier__n_estimators": [50, 100],
         "randomforestclassifier__max_depth": [10, 20]},
    ),
}

# 6. Tune each model with GridSearchCV and keep the best one
best_model = None
best_name = ""
best_score = 0

for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X_train, y_train)
    print(name, "| best settings:", grid.best_params_,
          "| cv accuracy:", round(grid.best_score_, 4))

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_name = name
        best_model = grid.best_estimator_

print("\nBest model:", best_name)

# 7. Evaluate the best model on the test data
pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# 8. Save the best model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("Saved model.pkl and encoders.pkl")
