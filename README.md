# 🏦 AI Loan Approval Prediction System

An end-to-end Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Google Gemini AI**. The application predicts whether a loan application will be approved or rejected based on applicant financial metrics and credit profile, and leverages Google Gemini to generate clear, plain-English explanations with recommendations.

---

## 📌 Features

- **Accurate Machine Learning Predictions**: Built using a Random Forest Classifier trained on financial assets, CIBIL score, income, and loan parameters.
- **Interactive Web Interface**: Clean, responsive UI with Streamlit.
- **AI-Powered Explanations**: Integrates Google Gemini (`gemini-1.5-flash`) to explain why a decision was made and provide actionable recommendations.
- **Production-Ready**: Easily deployable to Streamlit Community Cloud and GitHub.

---

## 📁 Project Structure

```
loan-approval-system/
│
├── loan_approval_dataset.csv     # Dataset (4,269 records)
├── train.py                     # Script to train ML model & save artifacts
├── predictor.py                 # Prediction helper module
├── app.py                       # Main Streamlit web application
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to exclude from Git
│
├── .streamlit/
│   └── secrets.toml.example     # Example Gemini API key configuration
│
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Generates `.pkl` files)
```bash
python train.py
```
This will train the Random Forest model and generate:
- `loan_approval_model.pkl`
- `label_encoders.pkl`
- `feature_names.pkl`

### 5. Configure Gemini API Key (Optional, for AI explanations)
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a file named `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```

### 6. Run the Streamlit Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push this project to your GitHub account (instructions below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Under **Advanced settings -> Secrets**, add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
6. Click **Deploy!** Your app will be live on the web with a public URL!

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn (Random Forest Classifier)
- **Data Manipulation**: Pandas, NumPy
- **Model Serialization**: Joblib
- **Generative AI**: Google Generative AI (Gemini 1.5 Flash)
