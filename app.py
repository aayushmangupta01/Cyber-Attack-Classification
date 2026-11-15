import streamlit as st
import pandas as pd
import pickle

st.title("Cyber Attack Classification")
st.write("Upload a CSV file of network traffic to find attacks.")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

file = st.file_uploader("Upload CSV", type="csv")

if file is not None:
    data = pd.read_csv(file)

    # remove the answer column if the file has it
    if "class" in data.columns:
        data = data.drop("class", axis=1)

    df = data.copy()

    # convert text to numbers (unknown values become -1)
    for col in encoders:
        mapping = {v: i for i, v in enumerate(encoders[col].classes_)}
        df[col] = df[col].map(mapping).fillna(-1)

    data["prediction"] = model.predict(df)

    st.write(data["prediction"].value_counts())
    st.dataframe(data.head(20))
