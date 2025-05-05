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
import pandas as pd
df=pd.read_csv('C:\\Users\\corne\\Dropbox\\PythonScripts\\BSS Retail Data.csv')
df
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
# Combine into a DataFrame for easy viewing
missing_data = pd.DataFrame({
'Missing Values': missing_counts,
'Missing (%)': missing_percent.round(2)
})
# Filter only columns with missing values
missing_data = missing_data[missing_data['Missing Values'] > 0]
# Display the result sorted by highest missing %
missing_data.sort_values(by='Missing (%)', ascending=False)
#deop column comp_5_price
df = df.drop(columns=['comp_5_price'])
df
#drop column comp_4_price
df = df.drop(columns=['comp_4_price'])
df
df['comp_1_price'] = df['comp_1_price'].fillna(df['comp_1_price'].mean())
df['managed_fba_stock_level'] = df['managed_fba_stock_level'].ffill()
# forward fill
df['comp_2_price_missing'] = df['comp_2_price'].isnull().astype(int)
df
# Select numeric columns
numeric_cols = df.select_dtypes(include='number').columns
# Create a function to identify outliers
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
return pd.DataFrame(outlier_summary).T.sort_values(by='Total Outliers',
ascending=False)
# Run the outlier detection
outlier_results = detect_outliers_iqr(df, numeric_cols)
# Display results
outlier_results
#Capping and Removing Outliers
def cap_outliers(df, column):
lower = df[column].quantile(0.05)
upper = df[column].quantile(0.95)
df[column] = df[column].clip(lower, upper)
columns_to_cap = ['sales', 'profit', 'price']
for col in columns_to_cap:
cap_outliers(df, col)
#Remove Outliers:
def remove_outliers_iqr(df, column):
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
return df[(df[column] >= lower) & (df[column] <= upper)]
columns_to_filter = ['unitsordered', 'cogs']
for col in columns_to_filter:
df = remove_outliers_iqr(df, col)
df.columns
df.head()
# Load the dataset
data = df.copy()
# Convert salesdate to datetime
data['salesdate'] = pd.to_datetime(data['salesdate'])
# Create a week number column for aggregation
data['week'] = data['salesdate'].dt.isocalendar().week
data['year'] = data['salesdate'].dt.year
data['year_week'] = data['year'].astype(str) + '-' + data['week'].astype(str)
# App Title and Description
st.title("Interactive Retail Data Pattern Exploration App")
st.write("Explore patterns in the retail sales data")
# Sidebar Filters
st.sidebar.header("Filter Options")
# Sidebar Filters for Date Range
min_date = data['salesdate'].min().date()
max_date = data['salesdate'].max().date()
date_range = st.sidebar.date_input(
"Sales Date Range",
[min_date, max_date]
)
# Convert selected dates to datetime
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
# Sidebar Filters for Numerical Variables
units_range = st.sidebar.slider(
"Units Ordered Range",
int(data['unitsordered'].min()),
int(data['unitsordered'].max()),
(1, 100)
)
price_range = st.sidebar.slider(
"Price Range ($)",
float(data['price'].min()),
float(data['price'].max()),
(data['price'].min(), data['price'].max())
)
# Sidebar Filters for Categorical Variables
sku_options = st.sidebar.multiselect(
"Product SKU",
options=data['sku'].unique(),
default=data['sku'].unique()[:5] # Default to first 5 SKUs to avoid
overloading
)
# Filter data based on selections
filtered_data = data[
(data['salesdate'].between(start_date, end_date)) &
(data['unitsordered'].between(*units_range)) &
(data['price'].between(*price_range)) &
(data['sku'].isin(sku_options))
]
# Show filtered data if user selects the option
if st.sidebar.checkbox("Show Filtered Data"):
st.write(filtered_data)
# Weekly Sales Histogram
st.header("Distribution of Sales by Week")
st.write("This histogram shows the weekly sales in the filtered data.")
# Aggregate sales by week
weekly_sales = filtered_data.groupby('year_week')['sales'].sum().reset_index()
# Plot histogram
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='year_week', y='sales', data=weekly_sales, color='skyblue', ax=ax)
ax.set_title("Histogram of Weekly Sales")
ax.set_xlabel("Year-Week")
ax.set_ylabel("Total Sales ($)")
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
st.pyplot(fig)
# Sales vs SKU
st.header("Sales by Product SKU")
st.write("This histogram shows the relationship between products (SKUs) and sales.")
# Aggregate data by SKU
sku_sales = filtered_data.groupby('sku')['sales'].sum().reset_index()
# Create a categorical plot for SKU vs Sales
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='sku', y='sales', data=sku_sales, ax=ax)
ax.set_title("Sales by Product SKU")
ax.set_xlabel("Product SKU")
ax.set_ylabel("Total Sales ($)")
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
st.pyplot(fig)
# Boxplot of Numerical Variables
st.header("Boxplot of Key Numerical Variables")
st.write("This boxplot shows the distribution of key numerical variables in the filtered data.")
# Select variables for boxplot
boxplot_vars = ['price', 'sales', 'profit', 'unitsordered', 'cogs']
boxplot_vars = [var for var in boxplot_vars if var in filtered_data.columns]
# Create a melted DataFrame for easier plotting
melted_data = pd.melt(filtered_data[boxplot_vars])
# Create the boxplot
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='variable', y='value', data=melted_data, ax=ax)
ax.set_title("Boxplot of Key Variables")
ax.set_xlabel("Variables")
ax.set_ylabel("Values")
plt.xticks(rotation=45)
st.pyplot(fig)
# Pairplot of Selected Variables
st.header("Pairplot of Key Variables")
st.write("This pairplot shows the relationships between key variables.")
# Select variables for pairplot (limited to avoid overplotting)
pairplot_vars = ['price', 'sales', 'profit', 'unitsordered']
pairplot_vars = [var for var in pairplot_vars if var in filtered_data.columns]
# Create sample if dataset is too large to avoid long rendering times
sample_size = min(1000, len(filtered_data))
sample_data = filtered_data[pairplot_vars].sample(sample_size, random_state=42)
# Create the pairplot
fig = plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(sample_data, diag_kind='kde')
plt.tight_layout()
st.pyplot(pairplot.fig)
# Correlation Matrix
st.header("Correlation Matrix")
st.write("Check the box to view the correlation matrix for numerical variables.")
# List of numerical variables for correlation
numerical_vars = ['price', 'unitsordered', 'sales', 'cogs', 'fba',
'reffee', 'adspend', 'profit', 'comp_1_price',
'comp_data_min_price', 'comp_data_max_price']
# Filter out only numerical columns that exist in the dataset
available_vars = [var for var in numerical_vars if var in filtered_data.columns]
# Show correlation matrix
if st.checkbox("Show Correlation Matrix"):
corr_matrix = filtered_data[available_vars].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
# Adding a Scatter Plot with trend line option
st.header("Scatter Plot: Price vs. Sales")
st.write("Check the box below to add a trendline to the scatter plot.")
show_trendline = st.checkbox("Show Trendline", value=False)
fig = px.scatter(filtered_data, x='price', y='sales', title="Price vs. Sales",
labels={"price": "Price ($)", "sales": "Sales ($)"},
trendline="ols" if show_trendline else None)
st.plotly_chart(fig)
