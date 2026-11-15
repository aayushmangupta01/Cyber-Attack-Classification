# Cyber Attack Classification

A simple machine learning project that classifies network traffic as
**normal** or **anomaly** (attack) using the NSL-KDD dataset.

## Files
| File | Purpose |
|------|---------|
| train.csv | dataset (41 features + `class` column) |
| train.py | trains a Random Forest, saves `model.pkl` and `encoders.pkl` |
| test.py | checks the saved model's accuracy |
| app.py | Streamlit app: upload a CSV and get predictions |
| .github/workflows/ci.yml | runs train + test on every push |

## How to run (Mac)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python test.py
streamlit run app.py
```
In the app, upload `train.csv` (or any CSV with the same columns).
