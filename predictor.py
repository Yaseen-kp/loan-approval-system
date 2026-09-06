import joblib
import pandas as pd
import os

# Load saved artifacts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_approval_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

model = None
encoders = None
feature_names = None

def load_artifacts():
    global model, encoders, feature_names
    if model is None:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

def predict_loan(user_data):
    """
    Predict loan approval from user input.

    Parameters:
        user_data (dict): User input from Streamlit.

    Returns:
        prediction (str): 'Approved' or 'Rejected'
        probabilities (dict): Probability for each class
    """
    load_artifacts()

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])

    # Encode categorical columns
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Prediction
    prediction = model.predict(input_df)[0]

    # Probabilities
    probability = model.predict_proba(input_df)[0]
    probability_dict = dict(zip(model.classes_, probability))

    return prediction, probability_dict
