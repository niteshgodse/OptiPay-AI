# OptiPay AI

OptiPay AI is an ML-powered smart payment recommendation system that predicts the best payment method for maximum cashback and savings.

Built using Python, Streamlit, and Random Forest Regression, the system analyzes transaction amount, platform, category, bank offers, and payment methods to recommend the most rewarding payment option in real time.

---

## Features

- Smart cashback prediction using Machine Learning
- Real-time payment recommendation engine
- Interactive Streamlit UI
- Bank-wise offer comparison
- Savings percentage analysis
- Dynamic recommendation explanation box
- ML model integration using PKL files

---

## Project Files Structure

```
OptiPay_AI/
│
├── app.py
├── OptiPay_AI.ipynb
├── best_model.pkl
├── label_encoders.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Joblib
- Matplotlib

---

## Files Included

- `app.py` → Main Streamlit application
- `OptiPay_AI.ipynb` → Model training notebook
- `best_model.pkl` → Trained ML model
- `label_encoders.pkl` → Saved label encoders
- `requirements.txt` → Project dependencies

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
