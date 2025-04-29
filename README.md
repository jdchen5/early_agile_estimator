# 🧠 Agile Project Estimator (Streamlit App)

This Streamlit application uses machine learning to provide **early effort estimations (in man-months)** for Agile software projects. It allows users to input project parameters and select a trained model to predict how much time the project may take.

---

## 🚀 Features

- Estimate man-months using:
  - **Linear Regression**
  - **Random Forest Regressor**
- Real-time prediction based on:
  - Project Complexity
  - Team Experience
  - Number of Requirements
  - Team Size
  - Technology Stack Complexity
- Optional: Generate sample models if none exist
- Modular and clean structure

---

## 📁 Folder Structure
agile_estimator_app/
├── main.py                  # Entry point of the Streamlit app
├── models.py                # Model creation, loading, prediction logic
├── ui.py                    # Streamlit UI components
├── models/                  # Folder where pickled models and scaler are saved
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── requirements.txt         # Python dependencies for the app
└── README.md                # Documentation on how to run and use the app

---

## 🛠️ Setup Instructions

1. **Clone the repo** or copy the files into a folder:
   ```bash
   git clone https://github.com/yourname/agile-estimator.git
   cd agile-estimator
2. Install dependencies:
    pip install -r requirements.txt
3. Run the app:
    streamlit run main.py

