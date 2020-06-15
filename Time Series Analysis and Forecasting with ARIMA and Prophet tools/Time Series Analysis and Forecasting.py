#Time Series Analysis and Forecasting with Python
#Tutorial Link: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

#------------------------------------------------------------------------------------------------------------------
#import packages
#------------------------------------------------------------------------------------------------------------------
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm
import matplotlib
import warnings
import itertools
from pylab import rcParams
import fbprophet
from fbprophet import Prophet

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

#set figure parameters
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

#------------------------------------------------------------------------------------------------------------------
#Import data, and preprocessing
#------------------------------------------------------------------------------------------------------------------
#read in data
path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data/"
superstores_data = pd.read_csv(path+"Superstores.csv")
superstores_data.head()

#lets focus on just furniture sales
furniture_data = superstores_data[superstores_data["Category"] == "Furniture"]; furniture_data.head()
furniture_data.isnull().sum()  #no missing data 
#min and max time of orders
print("Earliest order date: ", min(furniture_data["Order Date"]), ". Latest order date: ", max(furniture_data["Order Date"]))

#preprocessing furniture sales data for analysis
furniture_sales_data = furniture_data[["Order Date", "Sales"]]
furniture_sales_data = furniture_sales_data.sort_values(by = ["Order Date"])
furniture_sales_data.isnull().sum()

#group sales by date, and calculate average daily sales for each month - then set the date as an index (datetime format)
furniture_sales_data = furniture_sales_data.groupby("Order Date")['Sales'].sum().reset_index()
furniture_sales_data = furniture_sales_data.set_index('Order Date')

#average daily sales
furniture_sales_data.index = pd.to_datetime(furniture_sales_data.index)
furniture_sales_monthly_data = furniture_sales_data["Sales"].resample("MS").mean()
furniture_sales_monthly_data.head()

#------------------------------------------------------------------------------------------------------------------
#Exploratory visulaizations
#------------------------------------------------------------------------------------------------------------------
#visualizing the data
furniture_sales_monthly_data.plot(figsize = (15, 6))
plt.show()
#Here is where it gets interesting:
#We observe seasonality in our data - ups and lows 
#We can break down our time series into 3 components using the sm module's time series decomposition - trend, seasonality, and noise
rcParams['figure.figsize'] = 18, 8
decompose = sm.tsa.seasonal_decompose(furniture_sales_monthly_data, model='additive')
decompose.plot()
plt.show()  #furniture sales are pretty unstable, observe trend and seaonality

#------------------------------------------------------------------------------------------------------------------
#Time Series Model - ARIMA Forecast
#------------------------------------------------------------------------------------------------------------------
"""Autoregressive Integrated Moving Average
Model takes 3 parameters that account for seasonality, trend and noise
Notation: ARIMA(p, d, q)
Format= sm.tsa.statespace.SARIMAX(y,
                                order=(p, d, q),
                                seasonal_order=(4 paramters a combinations of pdq),
                                enforce_stationarity=False,
                                enforce_invertibility=False)"""
#parameter combination calculation
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#create function to run all possible combinations and then select the combination with the lowest AIC to use in our model

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(furniture_sales_monthly_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

print("Combination with lowest AIC: ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:297.78754395465455")

#fit the model - create model parameters, initialize the model, and then fit the model
model_parameters = sm.tsa.statespace.SARIMAX(furniture_sales_monthly_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
ARIMA_model = model_parameters.fit()
ARIMA_model.summary()
#diagnostics on the model
ARIMA_model.plot_diagnostics(figsize = (16, 8))
plt.show()   #correlogram is a plot of autocorrelation - visual way to show serial correlation in data that changes over time

#------------------------------------------------------------------------------------------------------------------
#Validating the ARIMA model
#------------------------------------------------------------------------------------------------------------------
#validate model - predict values and compare to existing ones
predictions = ARIMA_model.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
predictions_confidence_int = predictions.conf_int() #used to show variation for actual predicted sales

fig_prediction = furniture_sales_monthly_data['2014':].plot(label = 'Observed')
predictions.predicted_mean.plot(ax=fig_prediction, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
fig_prediction.fill_between(predictions_confidence_int.index,
                predictions_confidence_int.iloc[:, 0],
                predictions_confidence_int.iloc[:, 1], color='k', alpha=.2)
fig_prediction.set_xlabel('Date')
fig_prediction.set_ylabel('Furniture Sales')
plt.legend()  #activates the labels 
plt.show()

#validate the predictions statistically - using mean squared error and root mean squared error 
forecast = predictions.predicted_mean
truth = furniture_sales_monthly_data['2017-01-01':]   #starts 2017-01-01

mean_squared_error = ((forecast - truth)**2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mean_squared_error, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mean_squared_error), 2))) #we are able to predict sales to within 151.54 of the sales which range from 400 -1200

#------------------------------------------------------------------------------------------------------------------
#Make predictions or forecast of future furniture sales
#------------------------------------------------------------------------------------------------------------------
predictions_future = ARIMA_model.get_forecast(steps = 100)  #steps dictates how many steps into the future one would like to predict
predictions_future_confidence_int = predictions_future.conf_int()

fig_future_predictions = furniture_sales_monthly_data.plot(label='Observed', figsize=(15, 8))
predictions_future.predicted_mean.plot(ax=fig_future_predictions, label='Forecast')
fig_future_predictions.fill_between(predictions_future_confidence_int.index,
                predictions_future_confidence_int.iloc[:, 0],
                predictions_future_confidence_int.iloc[:, 1], color='k', alpha=.25)
fig_future_predictions.set_xlabel('Date')
fig_future_predictions.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

#------------------------------------------------------------------------------------------------------------------
"""Comparative Forecast
For example: Office Supplies vs Technology"""
#------------------------------------------------------------------------------------------------------------------
#create sub-datasets
office_data = superstores_data[superstores_data["Category"] == "Office Supplies"]
technology_data = superstores_data[superstores_data["Category"] == "Technology"]
furniture_data = superstores_data[superstores_data["Category"] == "Furniture"]
#create function to clean and preprocess data 
def clean_data(data):
    print("Data Shape: ", data.shape)
    #select order date and sales columns
    sales_data = data[["Order Date", "Sales"]]
    #sort data by order date
    sales_data = sales_data.sort_values(by = ["Order Date"])
    print("Missing data :", sales_data.isnull().sum())
    #group sales by order date and set date as index
    sales_daily_data = sales_data.groupby('Order Date')['Sales'].sum().reset_index()
    sales_daily_data = sales_daily_data.set_index('Order Date')
    sales_daily_data.index = pd.to_datetime(sales_daily_data.index)
    #mean sales per month
    return sales_daily_data['Sales'].resample('MS').mean()

office_monthly_data = clean_data(office_data); office_monthly_data.head()
technology_monthly_data = clean_data(technology_data); technology_monthly_data.head()
#concatanate into one dataframe
office_technology_monthly_data = pd.DataFrame({'Order Date':office_monthly_data.index, 'Sales':office_monthly_data.values}).merge(
    pd.DataFrame({'Order Date': technology_monthly_data.index, 'Sales': technology_monthly_data.values}), how='inner', on='Order Date')
office_technology_monthly_data.rename(columns={'Sales_x': 'Office_Sales', 'Sales_y': 'Technology_Sales'}, inplace=True)
office_technology_monthly_data.head()

#------------------------------------------------------------------------------------------------------------------
#Exploratory data visualization
#------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(20, 8))
plt.plot(office_technology_monthly_data["Order Date"], office_technology_monthly_data["Office_Sales"], '-g', label = "Office")
plt.plot(office_technology_monthly_data["Order Date"], office_technology_monthly_data["Technology_Sales"], '-b', label = "Technology")
plt.xlabel("Order Date"); plt.ylabel("Sales")
plt.legend(); plt.show()

#------------------------------------------------------------------------------------------------------------------
#Comparative Forecasting with PROPHET - Facebook's Time Series Tool
#------------------------------------------------------------------------------------------------------------------
"""Prophet packages works with Date as ds and the primary variable as y hence we have to rename our columns"""
#rename columns
office_monthly_data = pd.DataFrame({'Order Date':office_monthly_data.index, 'Sales':office_monthly_data.values})
office_monthly_data = office_monthly_data.rename(columns = {"Order Date": "ds", "Sales": "y"})
technology_monthly_data = pd.DataFrame({'Order Date': technology_monthly_data.index, 'Sales': technology_monthly_data.values})
technology_monthly_data = technology_monthly_data.rename(columns = {"Order Date": "ds", "Sales": "y"})
#------------------------------------------------------------------------------------------------------------------
#Run models
#------------------------------------------------------------------------------------------------------------------
#initialize the Prophet models
office_model = Prophet(interval_width=0.95).fit(office_monthly_data)
technology_model = Prophet(interval_width=0.95).fit(technology_monthly_data)

#create forecasts
office_forecast = office_model.make_future_dataframe(periods=36, freq='MS')
office_forecast = office_model.predict(office_forecast)

technology_forecast = technology_model.make_future_dataframe(periods=36, freq='MS')
technology_forecast = technology_model.predict(technology_forecast)

#create visualization of predictions
#start with office - remember, data ends in 2017, so black data points are actual data points, and after 12/31/2017, its forecasting
#office forecast
plt.figure(figsize=(18, 6))
office_model.plot(office_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Office Supplies Sales')
#office forecast components
office_model.plot_components(office_forecast)

#technology forecast
plt.figure(figsize=(18, 6))
technology_model.plot(technology_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Technology Supplies Sales')
#technology components
technology_model.plot_components(technology_forecast)

#plot forecasts together
#merge the 2 forecast tables and clean data 
office_names = ['office_%s' % column for column in office_forecast.columns]
technology_names = ['technology_%s' % column for column in technology_forecast.columns]

merge_office_forecast = office_forecast.copy()
merge_technology_forecast = technology_forecast.copy()

merge_office_forecast.columns = office_names
merge_technology_forecast.columns = technology_names

office_technology_forecast = pd.merge(merge_office_forecast, merge_technology_forecast, how = 'inner', left_on = 'office_ds', right_on = 'technology_ds')
office_technology_forecast = office_technology_forecast.rename(columns={'office_ds': 'Date'}).drop('technology_ds', axis=1)
office_technology_forecast.head()
#create comparative forecast estimates
plt.figure(figsize=(15, 8))
plt.plot(office_technology_forecast['Date'], office_technology_forecast['office_yhat'], 'g-', label = "Office Estimate")
plt.plot(office_technology_forecast['Date'], office_technology_forecast['technology_yhat'], 'b-', label = "Technology Estimate")
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Office vs. Technology Supplies Estimate')


