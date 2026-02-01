# ================================
# Wine Quality Prediction App
# EDA + Streamlit Dashboard + ML
# ================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ================================
# Page Config
# ================================
st.set_page_config(page_title="Wine Quality Analysis", layout="wide")

st.title("🍷 Wine Quality Analysis & Prediction Dashboard")

# ================================
# Load Data
# ================================
@st.cache_data
def load_data():
    red = pd.read_csv("data/winequality-red.csv", sep=";")
    white = pd.read_csv("data/winequality-white.csv", sep=";")

    red["color"] = "red"
    white["color"] = "white"

    df = pd.concat([red, white], ignore_index=True)
    return df

df = load_data()

# ================================
# Sidebar Filters
# ================================
st.sidebar.header("Filter Options")

color_filter = st.sidebar.multiselect(
    "Select Wine Color",
    df["color"].unique(),
    default=df["color"].unique()
)

alcohol_range = st.sidebar.slider(
    "Alcohol Range",
    float(df["alcohol"].min()),
    float(df["alcohol"].max()),
    (float(df["alcohol"].min()), float(df["alcohol"].max()))
)

filtered_df = df[
    (df["color"].isin(color_filter)) &
    (df["alcohol"].between(alcohol_range[0], alcohol_range[1]))
]

# ================================
# Dataset Overview
# ================================
st.subheader("📊 Dataset Overview")
st.write("Shape of dataset:", filtered_df.shape)
st.dataframe(filtered_df.head())

# ================================
# EDA Section
# ================================
st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Wine Quality Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["quality"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Alcohol vs Quality")
    fig, ax = plt.subplots()
    sns.boxplot(x="quality", y="alcohol", data=filtered_df, ax=ax)
    st.pyplot(fig)

# ================================
# Correlation Heatmap
# ================================
st.write("### 🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    filtered_df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm",
    annot=False,
    ax=ax
)
st.pyplot(fig)

