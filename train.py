import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv("loan_approval_dataset.csv")

    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()

    # Strip whitespace from string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Target and Features
    target_col = "loan_status"
    feature_cols = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categorical features
    encoders = {}
    categorical_cols = ["education", "self_employed"]

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"Encoded '{col}': classes = {le.classes_}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    print("Saving model artifacts...")
    joblib.dump(model, "loan_approval_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    joblib.dump(feature_cols, "feature_names.pkl")

    print("\n[SUCCESS] Successfully saved:")
    print(" - loan_approval_model.pkl")
    print(" - label_encoders.pkl")
    print(" - feature_names.pkl")

if __name__ == "__main__":
    main()
