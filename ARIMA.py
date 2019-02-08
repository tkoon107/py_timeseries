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


#------------------------------------------------------------------------
'''
Todo: Create file window or pull from URL
      Think of ways of logging output in a meaningful fashion
      Try additional transformations and preprocessing

      import Tkinter,tkFileDialog
        root = Tkinter.Tk()
        file = tkFileDialog.askopenfile(parent=root,mode='rb',title='Choose a file')
        if file != None:
            data = file.read()
            file.close()
            print "I got %d bytes from this file." % len(data)

'''






#------------------------------------------------------------------------
os.chdir(r"C:\Users\Trevor\Google Drive\docs\website\python time series")
data = pd.read_csv(r'ADM_problem_set_dataset.csv', parse_dates=['Date'], index_col='Date')

#Add daily frequency to time_index to create timeseries for each product
prev_index = data.index
time_index = pd.date_range(start=prev_index[0].strftime('%m/%d/%Y'), periods=len(prev_index), freq='D')
days = list(range(len(time_index)))

product_array = np.nan_to_num(data.values)
#products_array = data.values[data.values == 0] = min(data.values[data.values != 0])

series_dict = {}

#Create  a series object for each product
i = 0

for product in product_array.T: 
    median_val = np.median(product[product != 0])
    series_dict[i] = pd.Series(product, time_index)
    #replace zeroes with the min value of series
    series_dict[i].replace(0, median_val, inplace=True)
    i += 1
'''for product in product_array.T:
    series_dict[i] = pd.Series(product, time_index)
    series_dict[i][series_dict[i] == 0] = np.min(series_dict[i][series_dict[i] != 0])#Replace zeroes with min value of np.array
    series_dict[i]
    i += 1'''
    

    

#plot a few products
%matplotlib auto
plt.subplot(3, 1, 1)
plt.plot(series_dict[100])
plt.subplot(3, 1, 2)
plt.plot(series_dict[200])
plt.subplot(3, 1, 3)
plt.plot(series_dict[300])

#Construct model

#preprocessing
test_series = series_dict[0]

#Dickey-Fuller Test
dftest = adfuller(test_series, autolag='AIC')

test_series_detrend = test_series - pd.ewma(test_series, span=12)
dftest2 = adfuller(test_series_detrend, autolag='AIC')

test_series_ma_detrend = (test_series - pd.rolling_mean(test_series, 12)).dropna()
dftest3 = adfuller(test_series_ma_detrend, autolag='AIC')


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
lag_pacf = pacf(test_series, nlags=20, method='ols')
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
