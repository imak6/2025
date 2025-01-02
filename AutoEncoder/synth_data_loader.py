import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset():
    # Download latest version
    path = kagglehub.dataset_download("mtalaltariq/paysim-data")
    print("Path to dataset files:", path)
    df = pd.read_csv(path + '/paysim dataset.csv')
    return df

def preprocess_data(df):
    # Check if there are any NaN values in the data
    # df.isna()
    # df.loc[df.isna().any(axis="columns")]
    # # get the df info
    # df.info()
    # df.describe()
    # Drop unnecessary columns
    df = df.drop(columns=['isFlaggedFraud'])
    # Preprocess the data
    # Check for missing values
    # df.isnull().sum()

    # Fill missing values or drop rows/columns as appropriate
    df = df.fillna(0)  # Filling missing values with the mean (example)

    # df = df.sample(1000)
    # Standardize the numeric columns
    # Select only numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_cols]

    # Standardize the numeric columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Convert back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

    # # Concatenate scaled numeric features with non-numeric features if needed
    # df_final = pd.concat([df.drop(columns=numeric_cols), df_scaled], axis=1)

    return df_scaled