import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# Streamlit App Title
st.title("Exam 1 Part 2: Car Price Analysis")
st.write("This app analyzes the relationship between car characteristics and price using a cleaned automobile dataset.")

# Load Data
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/pb-7/exam1Piyush/main/Exam1_clean_df.csv"  # Replace with your GitHub CSV URL
    return pd.read_csv(url)

df = load_data()

# Show Dataset
st.header("Dataset Overview")
if st.checkbox("Show Dataset"):
    st.write(df)

# Data Types
st.subheader("Data Types")
if st.checkbox("Show Data Types"):
    st.write(df.dtypes)

# Section 1: Descriptive Statistics
st.header("Descriptive Statistics")
if st.checkbox("Show Descriptive Statistics"):
    st.write(df.describe())

# Section 2: Correlation Analysis
st.header("Correlation Analysis")
selected_columns = st.multiselect("Select columns for correlation matrix:", df.select_dtypes(include=['float64', 'int64']).columns)
if selected_columns:
    correlation = df[selected_columns].corr()
    st.write("Correlation Matrix:")
    st.write(correlation)
    st.write("Heatmap of Correlation Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section 3: Scatterplot Visualization
st.header("Scatterplot Visualization")
x_axis = st.selectbox("Select X-axis:", df.columns)
y_axis = st.selectbox("Select Y-axis:", df.columns)
if x_axis and y_axis:
    st.write(f"Scatterplot of {x_axis} vs {y_axis}:")
    fig, ax = plt.subplots()
    sns.regplot(x=x_axis, y=y_axis, data=df, ax=ax)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"Scatterplot of {x_axis} vs {y_axis}")
    st.pyplot(fig)

# Section 4: Boxplot Visualization
st.header("Boxplot Visualization")
category_col = st.selectbox("Select Categorical Column:", df.select_dtypes(include=["object"]).columns)
numeric_col = st.selectbox("Select Numeric Column:", df.select_dtypes(include=["float64", "int64"]).columns)
if category_col and numeric_col:
    st.write(f"Boxplot of {category_col} vs {numeric_col}:")
    fig, ax = plt.subplots()
    sns.boxplot(x=category_col, y=numeric_col, data=df, ax=ax)
    plt.xlabel(category_col)
    plt.ylabel(numeric_col)
    plt.title(f"Boxplot of {category_col} vs {numeric_col}")
    st.pyplot(fig)

# Section 5: Grouping and Pivot Table
st.header("Grouping and Pivot Table")
if st.checkbox("Show Grouping and Pivot Table Example"):
    df_group_one = df[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    st.write("Grouped Data (Drive Wheels and Body Style):")
    st.write(df_group_one)

    grouped_pivot = df_group_one.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
    st.write("Pivot Table:")
    st.write(grouped_pivot)

    st.write("Heatmap of Pivot Table:")
    fig, ax = plt.subplots()
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section 6: Hypothesis Testing
st.header("Hypothesis Testing")
numeric_col1 = st.selectbox("Select First Numeric Column for Hypothesis Test:", df.select_dtypes(include=["float64", "int64"]).columns)
numeric_col2 = st.selectbox("Select Second Numeric Column for Hypothesis Test:", df.select_dtypes(include=["float64", "int64"]).columns)
if numeric_col1 and numeric_col2:
    pearson_coef, p_value = stats.pearsonr(df[numeric_col1], df[numeric_col2])
    st.write(f"Pearson Correlation Coefficient between {numeric_col1} and {numeric_col2}: {pearson_coef:.2f}")
    st.write(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.success("The correlation is statistically significant.")
    else:
        st.warning("The correlation is not statistically significant.")

# Footer
st.write("---")
st.write("App created by Piyush Mohan Bhattarai.")
