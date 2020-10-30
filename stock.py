import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import math
import numpy as np

class stock:
    def __init__(self, file_name, title, train_size):
        self.file_name = file_name
        self.title = title
        self.train_size = train_size
        self.timeseries = pd.read_csv(file_name).reset_index()['Close']



    def adf_test(self, tseries):
        result = adfuller(tseries)
        labels = ['Augmented Dickey Fuller Test Statistics', 'p-value', 'Number of lags used', 'Number of observations used']

        for val, label in zip(result, labels):
            print(label + ' : ' + str(val))

        if result[1] <= 0.05:
            print("Reject null hypothesis, data is stationary")
        else:
            print("Accept null hypothesis, data is non-stationary")

    def test_stationarity(self):
        # Determing rolling statistics
        rolmean = self.timeseries.rolling(12).mean()
        rolstd = self.timeseries.rolling(12).std()
        # Plot rolling statistics:
        plt.plot(self.timeseries, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        # plt.show(block=False)

        self.adf_test(self.timeseries)

    def model_and_forecast(self):
        train_data = self.timeseries[0:(int(len(self.timeseries) * self.train_size)) - 1]
        test_data = self.timeseries[int(len(self.timeseries) * self.train_size) - 1:]
        print(int(len(self.timeseries) * self.train_size))
        df = pd.read_csv(self.file_name, usecols=[0, 4])
        df1 = pd.DataFrame(df, columns=['Date', 'Close'])
        df.columns = ["Date", "Close"]
        df = df['Date'] = pd.to_datetime(df['Date'])


        df1['Close'].shift(1)
        df['Close Values First Difference'] = df1['Close'] - df1['Close'].shift(1)
        df['Close Values First Difference'] = df['Close Values First Difference'].dropna()
        df['Seasonal First Difference'] = df1['Close'] - df1['Close'].shift(260)
        df['Seasonal First Difference'] = df['Seasonal First Difference'].dropna()

        plt.plot(df['Close Values First Difference'])
        plt.show()
        self.adf_test(df['Close Values First Difference'])

        result = seasonal_decompose(self.timeseries, model='multiplicative', freq=30)
        fig = plt.figure()
        fig = result.plot()
        fig.set_size_inches(16, 9)
        fig.show()

        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(self.timeseries)
        moving_avg = df_log.rolling(12).mean()
        std_dev = df_log.rolling(12).std()
        plt.legend(loc='best')
        plt.title('Moving Average')
        plt.plot(std_dev, color="black", label="Standard Deviation")
        plt.plot(moving_avg, color="red", label="Mean")
        plt.legend()
        plt.show()

        autocorrelation_plot(self.timeseries)
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = plot_acf(df['Close Values First Difference'].iloc[1:], lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = plot_pacf(df['Close Values First Difference'].iloc[0:], lags=40, ax=ax2)
        plt.show()

        model = ARIMA(train_data, order=(1, 1, 1))
        fitted = model.fit(disp=-1)
        print(fitted.summary())

        # Forecast
        fc, se, conf = fitted.forecast(31, alpha=0.05)
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)
        #plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(train_data[(len(self.timeseries) - 130):(len(self.timeseries) - len(fc_series))], label='Training')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        print(train_data)
        plt.plot(fc_series, color='orange', label='Predicted Stock Price')
        print('Forecasted: ')
        for i in range(31):
            print(fc_series[2493 + i])
        print('Lower Series')
        for i in range(31):
            print(lower_series[2493 + i])
        print('Upper Series')
        for i in range(31):
            print(upper_series[2493 + i])
        #plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
        plt.plot(lower_series, color = 'red', label='Lower Series')
        plt.plot(upper_series, color='green', label='Upper Series')
        plt.title(self.title + ' Stock Price Prediction')
        plt.xlabel('Business Days Since 11 Oct 2010')
        plt.ylabel('USD')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        plt.plot(train_data[(len(self.timeseries) - 130):(len(self.timeseries) - len(fc_series))], label='Training')
        plt.plot(test_data, color='blue', label='Actual Stock Price')
        plt.plot(fc_series, color='orange', label='Predicted Stock Price')
        plt.title(self.title + ' Stock Price Prediction')
        plt.xlabel('Business Days Since 11 Oct 2010')
        plt.ylabel('USD')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()