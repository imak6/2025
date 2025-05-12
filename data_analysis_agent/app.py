import streamlit as st
import pandas as pd
import numpy as np
import os
from main import setup_agent
# from analysis import main

st.set_page_config(page_title="Agent-Based Data Analyzer", layout="wide")
st.title("Agent-Based Data Analyzer")

uploaded_file = st.file_uploader("Upload a CSV/Excel/Parquet file", type=["csv", "xlsx", "parquet"])

if uploaded_file:
    temp_path = f"data/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success("File uploaded successfully! Running analysis...")

    agent = setup_agent()
    response = agent.invoke(f"Load the file at {temp_path} and summarize its structure, types, nulls, and key stats.")
    st.subheader("LLM-driven summary")
    st.text_area("LLM Output", response, height=250)

    viz_response = agent.invoke(f"generate and save visualization for the dataset")
    st.subheader("Visualizations")

    if os.path.exists("plots"):
        for file in os.listdir("plots"):
            if file.endswith(".png"):
                st.image(f"plots/{file}", caption=file)
