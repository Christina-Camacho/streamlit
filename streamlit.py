import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\corne\\Dropbox\\PythonScripts\\BSS Retail Data.csv')

# Identify numeric and categorical variables
numeric_vars = df.select_dtypes(include=['number']).columns.tolist()
categorical_vars = df.select_dtypes(include=['object']).columns.tolist()

# Print the results of Numeric Variables
print("\n Numeric Variables:")
for col in numeric_vars:
    print(f"- {col}")

# Print the results of Categorical Variables
print("\n Categorical Variables:")
for col in categorical_vars:
    print(f"- {col}")

# Calculate missingness
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_counts,
    'Missing (%)': missing_percent.round(2)
})
missing_data = missing_data[missing_data['Missing Values'] > 0]
missing_data.sort_values(by='Missing (%)', ascending=False)

# Drop columns
df = df.drop(columns=['comp_5_price', 'comp_4_price'])

# Impute and transform columns
df['comp_1_price'] = df['comp_1_price'].fillna(df['comp_1_price'].mean())
df['managed_fba_stock_level'] = df['managed_fba_stock_level'].ffill()
df['comp_2_price_missing'] = df['comp_2_price'].isnull().astype(int)

# Detect outliers
def detect_outliers_iqr(df, columns):
    outlier_summary = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            'Total Outliers': len(outliers),
            'Outlier %': round(100 * len(outliers) / len(df), 2),
            'Min Value': df[col].min(),
            'Max Value': df[col].max()
        }
    return pd.DataFrame(outlier_summary).T.sort_values(by='Total Outliers', ascending=False)

outlier_results = detect_outliers_iqr(df, df.select_dtypes(include='number').columns)

# Cap outliers
def cap_outliers(df, column):
    lower = df[column].quantile(0.05)
    upper = df[column].quantile(0.95)
    df[column] = df[column].clip(lower, upper)

for col in ['sales', 'profit', 'price']:
    cap_outliers(df, col)

# Remove outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in ['unitsordered', 'cogs']:
    df = remove_outliers_iqr(df, col)

# Prepare dataset
data = df.copy()
data['salesdate'] = pd.to_datetime(data['salesdate'])
data['week'] = data['salesdate'].dt.isocalendar().week
data['year'] = data['salesdate'].dt.year
data['year_week'] = data['year'].astype(str) + '-' + data['week'].astype(str)

# Streamlit app
st.title("Interactive Retail Data Pattern Exploration App")
st.write("Explore patterns in the retail sales data")

st.sidebar.header("Filter Options")
min_date = data['salesdate'].min().date()
max_date = data['salesdate'].max().date()
date_range = st.sidebar.date_input("Sales Date Range", [min_date, max_date])
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

units_range = st.sidebar.slider("Units Ordered Range",
    int(data['unitsordered'].min()),
    int(data['unitsordered'].max()),
    (1, 100))

price_range = st.sidebar.slider("Price Range ($)",
    float(data['price'].min()),
    float(data['price'].max()),
    (data['price'].min(), data['price'].max()))

sku_options = st.sidebar.multiselect("Product SKU",
    options=data['sku'].unique(),
    default=data['sku'].unique()[:5])

filtered_data = data[
    (data['salesdate'].between(start_date, end_date)) &
    (data['unitsordered'].between(*units_range)) &
    (data['price'].between(*price_range)) &
    (data['sku'].isin(sku_options))
]

if st.sidebar.checkbox("Show Filtered Data"):
    st.write(filtered_data)

st.header("Distribution of Sales by Week")
st.write("This histogram shows the weekly sales in the filtered data.")
weekly_sales = filtered_data.groupby('year_week')['sales'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='year_week', y='sales', data=weekly_sales, color='skyblue', ax=ax)
ax.set_title("Histogram of Weekly Sales")
ax.set_xlabel("Year-Week")
ax.set_ylabel("Total Sales ($)")
plt.xticks(rotation=90)
st.pyplot(fig)

st.header("Sales by Product SKU")
st.write("This histogram shows the relationship between products (SKUs) and sales.")
sku_sales = filtered_data.groupby('sku')['sales'].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='sku', y='sales', data=sku_sales, ax=ax)
ax.set_title("Sales by Product SKU")
ax.set_xlabel("Product SKU")
ax.set_ylabel("Total Sales ($)")
plt.xticks(rotation=90)
st.pyplot(fig)

st.header("Boxplot of Key Numerical Variables")
st.write("This boxplot shows the distribution of key numerical variables in the filtered data.")
boxplot_vars = ['price', 'sales', 'profit', 'unitsordered', 'cogs']
boxplot_vars = [var for var in boxplot_vars if var in filtered_data.columns]
melted_data = pd.melt(filtered_data[boxplot_vars])
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='variable', y='value', data=melted_data, ax=ax)
ax.set_title("Boxplot of Key Variables")
ax.set_xlabel("Variables")
ax.set_ylabel("Values")
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("Pairplot of Key Variables")
st.write("This pairplot shows the relationships between key variables.")
pairplot_vars = ['price', 'sales', 'profit', 'unitsordered']
pairplot_vars = [var for var in pairplot_vars if var in filtered_data.columns]
sample_size = min(1000, len(filtered_data))
sample_data = filtered_data[pairplot_vars].sample(sample_size, random_state=42)
fig = plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(sample_data, diag_kind='kde')
plt.tight_layout()
st.pyplot(pairplot.fig)

st.header("Correlation Matrix")
st.write("Check the box to view the correlation matrix for numerical variables.")
numerical_vars = ['price', 'unitsordered', 'sales', 'cogs', 'fba',
    'reffee', 'adspend', 'profit', 'comp_1_price',
    'comp_data_min_price', 'comp_data_max_price']
available_vars = [var for var in numerical_vars if var in filtered_data.columns]

if st.checkbox("Show Correlation Matrix"):
    corr_matrix = filtered_data[available_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.header("Scatter Plot: Price vs. Sales")
st.write("Check the box below to add a trendline to the scatter plot.")
show_trendline = st.checkbox("Show Trendline", value=False)
fig = px.scatter(filtered_data, x='price', y='sales', title="Price vs. Sales",
    labels={"price": "Price ($)", "sales": "Sales ($)"},
    trendline="ols" if show_trendline else None)
st.plotly_chart(fig)
