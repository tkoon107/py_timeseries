# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:08:51 2018

@author: Trevor
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.regression.linear_model import OLS
from datetime import datetime
from datetime import timedelta


#Grab data from github and parse date column
url = r'https://raw.githubusercontent.com/tkoon107/py_timeseries/master/ADM_problem_set_dataset.csv'
data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

#Add daily frequency to time_index to create timeseries for each product
prev_index = data.index
time_index = pd.date_range(start=prev_index[0].strftime('%m/%d/%Y'), periods=len(prev_index), freq='D')
product_array = data.values

series = pd.Series(product, time_index)
plt.plot(series)
#    series.replace(0, median_val, inplace=True)

#group by week
series_weeks = series.groupby(pd.Grouper(freq='W')).sum()[:-1] #omitting the last week as it only contained 2 days
series_weeks_test = series_weeks[-8:] #last 2 months which will be be predicted
series_weeks.drop(series_weeks.index[-8:], inplace=True)
plt.plot(series_weeks)

#Dickey-Fuller Test for stationality
#If critical value less than test statistic we fail to reject null hypothesis that series is NOT stationary
dftest = adfuller(series, autolag='AIC')
 
#using an estimated moving average to detrend the series
series_detrend =  series_weeks - series_weeks.ewm(halflife=8).mean()
plt.plot(series_detrend)

adfuller(series_detrend, autolag='AIC')
#Afer removing the exponentially weighted mean from each value the adfuller test shows we are 95% confident the series is stationary
#After reviewing the plot there does seem to be some seasonality to the dataset we can plot the variance across time to see this
series_detrend_std_deviation = series_detrend.rolling(window=8,center = False).std()


plt.plot(series_detrend, color = 'blue')
plt.plot(series_detrend_std_deviation ,color = 'orange')


#test diff
series_diff_detrend = (series_detrend - series_detrend.shift()).dropna()
series_diff_detrend_std_deviation = series_diff_detrend.rolling(window=8).std()

plt.plot(series_diff_detrend, color = 'blue')
plt.plot(series_diff_detrend_std_deviation ,color = 'orange')

adfuller(series_diff_detrend, autolag='AIC') #Well below 1% percentile critical value, will consider this to be stationary


#Forecast

arima_model = ARIMA(series_diff_detrend, order = (2, 1, 2))
results = arima_model.fit()
plt.plot(results.fittedvalues)
plt.plot(series_diff_detrend)

monthly_forecast = arima_model.forecast(steps=4)

#differencing
#ACF

critical_values = [-1.96/np.sqrt(len(test_series)), 1.96/np.sqrt(len(test_series))]
lag_acf = acf(test_series_detrend, nlags=20)
p = None
for i, lag in enumerate(lag_acf):
    print(i,lag)
    if lag >= critical_values[0] and lag <= critical_values[1]:
        p = i
        break
#PACF
lag_pacf = pacf(series, nlags=20, method='ols')
d = None
for i, lag in enumerate(lag_pacf):
    print(i,lag)
    if lag >= critical_values[0] and lag <= critical_values[1]:
        p = i
        break


#test_series_decomp = seasonal_decompose(test_series.asfreq('D').fillna(0))
#printout = test_series_decomp.plot()
#plot_mpl(printout)

new_date_range = pd.date_range(time_index[-1].to_pydatetime(), periods=30, freq='D')
start_index = days[3]
ending_index = days[-1]


arima_model = ARIMA(test_series_detrend, order = (19,1,3))
fitted_model = arima_model.fit(transparams = False)


monthly_forecast = fitted_model.forecast(steps=30)
plt.plot(monthly_forecast[0])
forecast_series = pd.Series(monthly_forecast, index = new_date_range)
plt.plot(forecast_series)


#INVERSE OUTPUT
inversed_back = monthly_forecast[0] + pd.ewma(test_series, halflife=12)
#SCALE FORECAST FOR EACH PRODUCT
predicted_values_dict = {}


for key, product in series_dict:
    product.replace(0, min([sales for sales in test_series if sales > 0]), inplace=True)
    arima_model = ARIMA(product, order = (3,1,3))
    fitted_model = arima_model.fit(transparams = False)
    predicted_values_dict[key] = fitted_model.forecast(steps=30)[0]

##################################################################

model = OLS(series_dict[1],days)
results = model.fit()
predictions = results.predict(range(755,755 + 72))
plt.plot(predictions)



from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(series_dict[1], freq = 365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

bacK_together = trend + seasonal + residual


from statsmodels.tsa.ar_model import AR
prev_index = data.index.asfreq('D')

model = AR(series_dict[1])
model_fit = model.fit()
model_predict = model_fit.


arima_model = ARIMA(series_dict[1], order = (3,1,3))
fitted_model = arima_model.fit()
forecasts = fitted_model.forecast(range(30))


    #cal = USFederalHolidayCalendar()
    #holidays = cal.holidays(start='2014-10-22', end='2017-12-31').to_pydatetime()
    #bday_cust = CDay(holidays=holidays, weekmask='Mon Tue Wed Thu Fri') 
    #time_index = pd.date_range(start=time_index[0].strftime('%m/%d/%Y'), periods=len(time_index), freq=bday_cust)
