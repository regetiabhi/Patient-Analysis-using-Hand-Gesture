import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. MODULE-2: PREPROCESSING
print("Loading data and preprocessing...")
df = pd.read_csv('data/gestures.csv')

# Check for missing values [cite: 158]
if df.isnull().values.any():
    df = df.dropna()

# Assign Independent (X) and Dependent (y) variables [cite: 156]
X = df.iloc[:, :-1].values  # All landmark coordinates
y = df.iloc[:, -1].values   # The labels (e.g., "Need_Water")

# Split data into training and testing [cite: 157]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODULE-3: MODEL TRAINING 
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_acc = 0
best_model = None
best_model_name = ""

print("\n--- PERFORMANCE EVALUATION ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}")
    # Display Precision, Recall, and F1-score [cite: 166-179]
    print(classification_report(y_test, predictions))
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

# 3. SAVE THE FINAL MODEL [cite: 67, 68]
print(f"\nSelecting {best_model_name} as the final model (Accuracy: {best_acc:.2f})")
joblib.dump(best_model, 'models/gesture_model.pkl')
print("Model saved to 'models/gesture_model.pkl'")