import streamlit as st
import pandas as pd
import os
from predictor import predict_loan

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* Hide Streamlit default menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Reduce top spacing */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

/* Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

/* Main title */
.main-title{
    text-align:center;
    font-size:52px;
    color:white;
    font-weight:bold;
    margin-bottom:5px;
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:20px;
    margin-bottom:40px;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:#2563eb;
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    background:#1d4ed8;
    transform:scale(1.02);
}

/* Input Boxes */
.stTextInput input,
.stNumberInput input{
    border-radius:12px !important;
}

/* Select Box */
.stSelectbox > div > div{
    border-radius:12px;
}

/* Headers */
h1,h2,h3,h4{
    color:white;
}

/* Success & Error Boxes */
.stSuccess,
.stError{
    border-radius:15px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    "<div class='main-title'>🏦 AI Loan Approval System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Machine Learning + Gemini AI Powered Loan Decision Assistant</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar: Settings & API Key
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input(
        "🔑 Gemini API Key (Optional)",
        type="password",
        help="Paste your API key here. It stays strictly in your browser session and is NEVER uploaded to GitHub."
    )
    st.markdown("---")
    st.caption("Don't have a key? Get one free at [Google AI Studio](https://aistudio.google.com/).")
    st.caption("The ML prediction model works completely offline even without an API key.")

# -----------------------------
# User Input Form
# -----------------------------
st.header("📝 Loan Application Form")

col1, col2 = st.columns(2)

with col1:
    # Number of Dependents
    no_of_dependents = st.selectbox(
        "Number of Dependents",
        [0, 1, 2, 3, 4, 5]
    )

    # Education
    education = st.radio(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    # Self Employed
    self_employed = st.radio(
        "Self Employed",
        ["Yes", "No"]
    )

    # Annual Income
    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        value=5000000,
        step=50000
    )

    # Loan Amount
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=15000000,
        step=50000
    )

    # Loan Term
    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        max_value=30,
        value=10
    )

with col2:
    # CIBIL Score
    cibil_score = st.number_input(
        "CIBIL Score (300 - 900)",
        min_value=300,
        max_value=900,
        value=750
    )

    # Residential Assets
    residential_assets_value = st.number_input(
        "Residential Assets Value (₹)",
        min_value=0,
        value=2500000,
        step=50000
    )

    # Commercial Assets
    commercial_assets_value = st.number_input(
        "Commercial Assets Value (₹)",
        min_value=0,
        value=1000000,
        step=50000
    )

    # Luxury Assets
    luxury_assets_value = st.number_input(
        "Luxury Assets Value (₹)",
        min_value=0,
        value=5000000,
        step=50000
    )

    # Bank Assets
    bank_asset_value = st.number_input(
        "Bank Assets Value (₹)",
        min_value=0,
        value=3000000,
        step=50000
    )

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Loan Approval"):

    # Create user data
    user_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    prediction, probability_dict = predict_loan(user_data)

    st.subheader("📊 Prediction Result")

    if prediction == "Approved":
        st.success(f"✅ Loan Approved (Confidence: {probability_dict.get('Approved', 1.0) * 100:.1f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {probability_dict.get('Rejected', 1.0) * 100:.1f}%)")

    # -----------------------------
    # GenAI Explanation
    # -----------------------------
    st.subheader("🤖 AI Explanation")

    # Priority: 1) User input in sidebar, 2) st.secrets, 3) environment variable
    gemini_key = None
    if user_api_key and user_api_key.strip():
        gemini_key = user_api_key.strip()
    elif "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ:
        gemini_key = os.environ["GEMINI_API_KEY"]

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)

            # Auto-detect the best available model for this API key
            selected_model_name = None
            try:
                available = [
                    m.name for m in genai.list_models()
                    if "generateContent" in getattr(m, "supported_generation_methods", [])
                ]
                # Prioritize flash models, then any available model
                flash_models = [m for m in available if "flash" in m.lower()]
                if flash_models:
                    selected_model_name = flash_models[0]
                elif available:
                    selected_model_name = available[0]
            except Exception:
                pass

            if not selected_model_name:
                selected_model_name = "gemini-1.5-flash-latest"

            model_ai = genai.GenerativeModel(selected_model_name)

            prompt = f"""
Loan Prediction: {prediction}

Applicant Details:
{user_data}

Explain in simple points in English:
1. Why the loan was approved/rejected.
2. Mention strong points if approved else mention weak points and suggest improvements.
dont say here is the AI explanation just say the above mentioned points,make the points as a list with single line points
"""

            with st.spinner("Generating AI explanation..."):
                response = model_ai.generate_content(prompt)
                st.write(response.text)

        except Exception as e:
            st.warning(f"Could not generate Gemini explanation: {e}")
    
