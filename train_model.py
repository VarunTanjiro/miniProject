import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("sign_data.csv")

print("Dataset loaded!")
print("Total samples:", len(data))

# Separate input and output
X = data.drop("label", axis=1)
y = data["label"]

print("Signs:", y.unique())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
print("\nTraining model...")
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "sign_model.pkl")

print("\nModel saved as sign_model.pkl")