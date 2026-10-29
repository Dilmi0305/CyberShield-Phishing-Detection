
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"

# Small educational dataset
# 0 = Legitimate, 1 = Phishing
data = [
    ("https://www.wikipedia.org", 0),
    ("https://www.python.org", 0),
    ("https://github.com", 0),
    ("https://www.microsoft.com", 0),
    ("https://www.google.com", 0),
    ("https://docs.python.org", 0),
    ("http://192.0.2.10/login/verify", 1),
    ("http://secure-login.example.invalid/account", 1),
    ("http://verify-account.example.invalid/update", 1),
    ("http://free-prize.example.invalid/claim", 1),
    ("http://password-reset.example.invalid/login", 1),
    ("http://bank-security.example.invalid/confirm", 1),
]

df = pd.DataFrame(data, columns=["url", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["url"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"],
)

model = Pipeline([
    (
        "features",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("CyberShield model training completed!")
print("Demo test accuracy:", round(
    accuracy_score(y_test, predictions), 2
))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
