# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:08:51 2018

@author: Trevor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf


#Grab data from github and parse date column
url = r'https://raw.githubusercontent.com/tkoon107/py_timeseries/master/ADM_problem_set_dataset.csv'
data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

#Add daily frequency to time_index to create timeseries for each product
prev_index = data.index
time_index = pd.date_range(start=prev_index[0].strftime('%m/%d/%Y'), periods=len(prev_index), freq='D')
product = data.values

series = pd.Series(product.T[0], time_index)
plt.plot(series)

#test 4

#    series.replace(0, median_val, inplace=True)
#group by week
series_weeks_full = series.groupby(pd.Grouper(freq='W')).sum()[:-1] #omitting the last week as it only contained 2 days
#last 2 months which will be be predicted series_weeks_test = series_weeks[-4:] 
series_weeks = series_weeks_full.drop(series_weeks_full.index[-4:], inplace=False)
plt.plot(series_weeks)

#Dickey-Fuller Test for statoinarity
#If critical value less than test statistic we fail to reject null hypothesis that series is NOT stationary
adfuller(series_weeks, autolag='AIC')
 
#using a log transformation and estimated moving average to detrend the series

series_log_detrend = np.log(series_weeks) - np.log(series_weeks).ewm(halflife=8).mean()
plt.plot(series_log_detrend)
adfuller(series_log_detrend)

series_detrend =  series_weeks - series_weeks.ewm(halflife=8).mean()
plt.plot(series_detrend)

adfuller(series_log_detrend, autolag='AIC')
#Afer removing the exponentially weighted mean from each value the adfuller test shows we are 95% confident the series is stationary
#After reviewing the plot there does seem to be some seasonality to the dataset we can plot the variance across time to see this
series_log_detrend_sd = series_log_detrend_sd.rolling(window=8,center = False).std()

plt.plot(series_log_detrend, color = 'blue')
plt.plot(series_log_detrend ,color = 'orange')

#test diff
series_diff_detrend = (series_detrend - series_detrend.shift()).dropna()
series_diff_detrend_std_deviation = series_diff_detrend.rolling(window=8).std()

plt.plot(series_diff_detrend, color = 'blue')
plt.plot(series_diff_detrend_std_deviation ,color = 'orange')

adfuller(series_diff_detrend, autolag='AIC') #Well below 1% percentile critical value, will consider this to be stationary

#Test autocorrelation between different lags with acf and pacf

lag_acf = acf(series_log_detrend, nlags=16)
lag_pacf = pacf(series_log_detrend, nlags=16)


#Plot ACF if a lag crosses critical value select as q parameter
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(series_log_detrend)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(series_log_detrend)),linestyle='--',color='gray')



#Plot PACF if a lag crosses critical value select as p parameter
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(series_log_detrend)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(series_log_detrend)),linestyle='--',color='gray')


#Forecast

arima_model = ARIMA(series_log_detrend, order = (1, 0, 1))
results = arima_model.fit()

plt.plot(series_log_detrend)
plt.plot(results.fittedvalues)

monthly_forecast = results.forecast(steps=4)



forecast_time_index = pd.date_range(series_log_detrend.index[-1] + pd.DateOffset(weeks=1), periods=4, freq='W')

#Reverse log transformation and ewa 
monthly_forecast = np.exp(monthly_forecast[0]) + series_weeks.ewm(halflife=8).mean()[-4:]

#Add forecast values to series
series_forecast_test = series_log_detrend.append(pd.Series(monthly_forecast, index = forecast_time_index))

#Compare forecasted series to actual series
plt.plot(forecasted_series)

#differencing
#ACF


#printout = test_series_decomp.plot()
#plot_mpl(printout)



#INVERSE OUTPUT
inversed_back = monthly_forecast[0] + pd.ewma(test_series, halflife=12)
