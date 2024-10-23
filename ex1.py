import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression #-m pip install scikit-learn

filename = "jbhunt_data.csv"
df = pd.read_csv(filename)
print(df.describe())
print(df.groupby('datadate').mean())

jbhunt_df = pd.read_csv(filename, index_col="datadate", infer_datetime_format=True)
jbhunt_df.index = pd.to_datetime(jbhunt_df.index)

# I am using 25 Years of Sales Data to forecase future sales.
# From this, I will forecast 12 years in the future from 2011 to 2022 and bounce that off of what we know Sales were during those 12 years.

# I need a Returns column to show me the sales and exclude any zero or NaN values, if they exist.
jbhunt_df['Return'] = (jbhunt_df["sales"])
returns = jbhunt_df.replace(-np.inf, np.nan).dropna()
returns = returns[returns['Return'] !=0]

# I need to train the basic regression model
train = jbhunt_df[:'2010-12-31']
test = jbhunt_df['2010-12-31':]
x_train = train["Return"].to_frame()
y_train = train["Return"]
x_test = test["Return"].to_frame()
y_test = test["Return"]

# At this point, I need to create a Linear Regression for this model fitted to the training data
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
print("Coefficients:", regression_model.coef_)
print("Intercept:", regression_model.intercept_)
predictions = regression_model.predict(x_test)
result = y_test.to_frame()
result['Predicted Return'] = predictions
plt.figure(figsize=(10, 13))
plt.plot(result.index[:13], result['Return'][:13], label='Actual Sales', marker='o')
for i, txt in enumerate(result['Return'][:13]):
    plt.annotate(f'{txt:.2f}', (result.index[i], result['Return'][i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Date')
plt.ylabel('Sales in Millions of Dollars')
plt.title('Actual Sales')
plt.legend(loc='upper left')
plt.show()

# In this second dataframe, I am using 12 years -- from 1998 to 2010 -- of JB Hunt Sales to build a test and training set.
# Following is using a Lagged Return prediction to establish the forecast model's accuracy now that we know the actual Sales for the 12 year timeframe we are using.
# I need a Returns column to show me the sales and exclude any zero or NaN values, if they exist.
jbhunt_df['Return'] = (jbhunt_df["sales"])
returns = jbhunt_df.replace(-np.inf, np.nan).dropna()
returns = returns[returns['Return'] !=0]

# I need lagged returns to serve me as a predictor for future returns on sales as I have learned in previous Finance and Firm Valuation courses.
jbhunt_df['Lagged_Return'] = jbhunt_df["Return"].shift()
jbhunt_df = jbhunt_df.dropna()

# I need to train the basic regression model
train = jbhunt_df[:'2010-12-31']
test = jbhunt_df['2010-12-31':]
x_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
x_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]

# At this point, I need to create a Linear Regression for this model fitted to the training data
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
print("Coefficients:", regression_model.coef_)
print("Intercept:", regression_model.intercept_)
predictions = regression_model.predict(x_test)
result = y_test.to_frame()
result['Predicted Return'] = predictions
plt.figure(figsize=(10, 13))
plt.plot(result.index[:13], result['Return'][:13], label='Actual Sales', marker='o')
for i, txt in enumerate(result['Return'][:13]):
    plt.annotate(f'{txt:.2f}', (result.index[i], result['Return'][i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.plot(result.index[:13], result['Predicted Return'][:13], label='Predicted Sales', marker='o')
for i, txt in enumerate(result['Predicted Return'][:13]):
    plt.annotate(f'{txt:.2f}', (result.index[i], result['Predicted Return'][i]), textcoords="offset points", xytext=(0,10), ha='left')
plt.xlabel('Date')
plt.ylabel('Sales in Millions of Dollars')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()


# Running an OLS model on this data set to extract the summary and run a different type of regression on the same data
train = jbhunt_df[:'2010-12-31']
test = jbhunt_df['2010-12-31':]
x = train['Lagged_Return']
y = train['Return']
x_constant = sm.add_constant(x)
model =sm.OLS(y,x_constant).fit()
x_ols_test = sm.add_constant(test["Lagged_Return"])
predictions = model.predict(x_ols_test)
plt.figure(figsize=(10, 13))
plt.plot(test.index[:13], test['Return'][:13], label='Actual Sales', marker='o')
plt.plot(test.index[:13], predictions[:13], label='Predicted Sales', marker='o')
for i, txt in enumerate(test['Return'][:13]):
    plt.annotate(f'{txt:.2f}', (test.index[i], test['Return'][i]), textcoords="offset points", xytext=(0,10), ha='center')
for i, txt in enumerate(predictions[:13]):
    plt.annotate(f'{txt:.2f}', (test.index[i], predictions[i]), textcoords="offset points", xytext=(0,10), ha='left')
plt.xlabel('Date')
plt.ylabel('Sales in Millions of Dollars')
plt.title('OLS Actual vs Predicted Sales')
plt.legend()
plt.show()
print(model.summary())