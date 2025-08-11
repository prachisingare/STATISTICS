# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set Streamlit page config
st.set_page_config(page_title="Store Revenue Insights", layout="wide")

# Title
st.title("📊 Store Revenue Insights")

# Generate dataset
np.random.seed(42)
data = {
    'product_id': range(1, 21),
    'product_name': [f'Product{i}' for i in range(1, 21)],
    'category': np.random.choice(['Electronic', 'Clothing', 'Home', 'Sports'], size=20),
    'units_sold': np.random.poisson(lam=20, size=20),
    'sales_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
}
sales_data = pd.DataFrame(data)

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dataset", "Descriptive Stats", "Visualizations", "Hypothesis Testing"])

# Dataset Page
if page == "Dataset":
    st.subheader("📂 Sales Dataset")
    st.dataframe(sales_data)
    csv = sales_data.to_csv(index=False)
    st.download_button("Download CSV", csv, "sales_data.csv", "text/csv")

# Descriptive Stats Page
elif page == "Descriptive Stats":
    st.subheader("📈 Descriptive Statistics")
    st.write(sales_data['units_sold'].describe())

    st.metric("Mean Units Sold", round(sales_data['units_sold'].mean(), 2))
    st.metric("Median Units Sold", sales_data['units_sold'].median())
    st.metric("Variance", round(sales_data['units_sold'].var(), 2))
    st.metric("Standard Deviation", round(sales_data['units_sold'].std(), 2))

    st.subheader("Category Statistics")
    st.dataframe(sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']))

# Visualization Page
elif page == "Visualizations":
    st.subheader("📊 Data Visualizations")

    plot_type = st.selectbox("Select Visualization", ["Histogram", "Boxplot", "Bar Plot"])

    if plot_type == "Histogram":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(sales_data['units_sold'], bins=10, kde=True, ax=ax)
        ax.set_title("Distribution of Units Sold")
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='category', y='units_sold', data=sales_data, ax=ax)
        ax.set_title("Boxplot of Units Sold by Category")
        st.pyplot(fig)

    elif plot_type == "Bar Plot":
        category_stats = sales_data.groupby("category")["units_sold"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='category', y='units_sold', data=category_stats, ax=ax)
        ax.set_title("Total Units Sold by Category")
        st.pyplot(fig)

# Hypothesis Testing Page
elif page == "Hypothesis Testing":
    st.subheader("📏 One Sample t-Test")
    mu = st.number_input("Enter hypothesized mean", value=20.0)
    t_stat, p_val = stats.ttest_1samp(sales_data['units_sold'], mu)

    st.write(f"T-statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        st.error("Reject the null hypothesis: Mean units sold is significantly different.")
    else:
        st.success("Fail to reject the null hypothesis: No significant difference found.")
