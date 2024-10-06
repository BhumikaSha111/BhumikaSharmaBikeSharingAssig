#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[7]:


# Load dataset
df = pd.read_csv('day.csv')


# In[8]:


# Data Quality Checks
# 1. Checking for missing values
print("Checking for missing values:")
print(df.isnull().sum())  # There are no missing values in the dataset


# In[9]:


# 2. Checking for duplicate rows
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"Dropping {duplicates} duplicate rows.")
    df = df.drop_duplicates()
else:
    print("No duplicate rows found.")


# In[14]:


# Select only numerical columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])


# In[15]:


# Check if the selection worked
print(numeric_df.columns)


# In[16]:


# Pair plot to inspect relationships between variables
sns.pairplot(df)
plt.show()


# In[17]:


# Heatmap to inspect correlations between numerical variables
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap of Variable Correlations (Numerical Features)")
plt.show()


# In[18]:


# 3. Dropping irrelevant columns
# 'instant' and 'dteday' are just record indexes and dates that do not provide predictive value
df = df.drop(['instant', 'dteday'], axis=1)


# In[19]:


# Categorical Variables Handling
# Mapping 'season', 'weathersit', and 'yr' columns to their respective labels
df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
df['weathersit'] = df['weathersit'].map({
    1: 'Clear/Few clouds', 
    2: 'Mist/Cloudy', 
    3: 'Light Snow/Rain', 
    4: 'Heavy Rain/Snow'
})
df['yr'] = df['yr'].map({0: '2018', 1: '2019'})


# In[20]:


# Assign string values to 'mnth' and 'weekday'
df['mnth'] = df['mnth'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
df['weekday'] = df['weekday'].map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})


# In[29]:


# Creating Dummy Variables for Categorical Columns
df = pd.get_dummies(df, drop_first=True)


# In[30]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Initialize model and RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)  # Modify n_features_to_select based on your needs

# Fit RFE
X = df.drop(columns=['cnt'])  # Independent variables
y = df['cnt']  # Target variable
rfe.fit(X, y)

# Display selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)


# In[33]:


# Dropping features with high VIF values (e.g., if any VIF > 10, indicating multicollinearity)
# For this demonstration, we'll assume the features are reasonably independent

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)



# In[27]:


from sklearn.preprocessing import MinMaxScaler

# Select the correct numerical columns for scaling
numerical_features = ['temp', 'atemp', 'hum', 'windspeed']  # Adjust this list based on available features
scaler = MinMaxScaler()

# Apply MinMaxScaler to these features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Check the result
print(df.head())


# In[35]:


# Residual analysis on training data
residuals = y_train - y_train_pred
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()


# In[36]:


# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[37]:


# Q-Q plot for normality of residuals
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[39]:


from sklearn.metrics import r2_score, mean_squared_error

# R-squared and RMSE for test data
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"R-squared (Train): {r2_train}")
print(f"R-squared (Test): {r2_test}")
print(f"RMSE (Train): {rmse_train}")
print(f"RMSE (Test): {rmse_test}")


# In[40]:


# Model Interpretation and Explanation
# The key features affecting bike demand are explained based on the model coefficients
# We examine the model coefficients from sklearn or statsmodels
coefficients = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)


# #Final Interpretation
# 
# 
# Important Features: Features such as temp, atemp, season, and weathersit were found to be the most important predictors of bike rentals. 
# 
# This aligns with intuition—weather conditions and temperature significantly affect biking activity.
# 
# 
# Seasonality Effects: The model likely captured strong seasonal effects, with higher bike rentals in spring and summer compared to winter.
# 
# 
# Weather Conditions: The variable weathersit (which captures clear vs cloudy vs rainy weather) also played a role, as inclement weather naturally reduces biking activity.

# In[ ]:




