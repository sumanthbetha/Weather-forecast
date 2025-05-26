import pandas as pd
import numpy as np

df=pd.read_csv(r'weatherBegumpet.csv')

dfw = df[['DATE', 'TAVG']]
dfw.isnull().sum()



import matplotlib.pyplot as plt

# Make sure 'DATE' is a datetime type
dfw['DATE'] = pd.to_datetime(dfw['DATE'], format='%Y-%m-%d')

# Set 'DATE' as index 
dfw.set_index('DATE', inplace=True)

dfw = dfw.asfreq('D')  
dfw.fillna(method='ffill', inplace=True)

dfw.plot(figsize=(10, 5), title='Tavg')
plt.show()




def mean_absolute_percentage_error(y_true, y_pred): 
    # Convert y_true and y_pred to numpy arrays for ease of computation
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Calculate absolute percentage error for each prediction
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    # Take the mean of all absolute percentage errors and multiply by 100 to get MAPE
    return np.mean(absolute_percentage_error) * 100

###################################### ARIMA ####################################################################




from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

split_index = int(len(dfw) * 0.8)

Train = dfw.iloc[:split_index]  
Test = dfw.iloc[split_index:]


# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(Train, lags=20, ax=plt.gca())  # ACF plot
plt.subplot(122)
plot_pacf(Train, lags=20, ax=plt.gca())  # PACF plot
plt.show()


from pmdarima import auto_arima

auto_model = auto_arima(Train, seasonal=False, trace=True)
print(auto_model.summary())


from statsmodels.tsa.arima.model import ARIMA




model = ARIMA(Train, order=(2,0,1))  # (p,d,q)
model_fit = model.fit()

forecast_steps = len(Test)
forecast1 = model_fit.forecast(steps=forecast_steps)

last_date = Train.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast1 = pd.Series(forecast1.values, index=forecast_index)


plt.plot(Test,label='Test Data')
plt.plot(forecast1, color='red')
plt.legend()

mape1 = mean_absolute_percentage_error(Test, forecast1)
mape1


######################################################### SARIMA #################################################



from pmdarima import auto_arima

auto_model = auto_arima(Train, 
                        seasonal=True, 
                        m=365,       # or 12 if monthly
                        trace=True,
                        stepwise=True,
                        suppress_warnings=True)

print(auto_model.summary())



from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(Train, order=(2, 0, 1), seasonal_order=(2, 0, 1, 365))  # Define seasonal order
fitted_model = model.fit()

forecast_steps = len(Test)
forecast2 = fitted_model.forecast(steps=forecast_steps)

plt.plot(Test,label='Test Data')
plt.plot(forecast2, color='red')
plt.legend()

mape2 = mean_absolute_percentage_error(Test, forecast2)
mape2

##########################################################  ETS #############################################

from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(Train, trend=None, seasonal='add', seasonal_periods=365)  # Example for monthly data
fitted_model = model.fit()

forecast3 = fitted_model.forecast(steps=forecast_steps)

plt.plot(Test,label='Test Data')
plt.plot(forecast3, color='red')
plt.legend()

mape3 = mean_absolute_percentage_error(Test, forecast3)
mape3

#########################################   Random Forest ##################################################
from sklearn.ensemble import RandomForestRegressor

dfr = dfw
dfr.reset_index(inplace=True)
dfr['DATE'] = pd.to_datetime(dfr['DATE'])



dfr['year'] = dfr['DATE'].dt.year
dfr['month'] = dfr['DATE'].dt.month
dfr['day'] = dfr['DATE'].dt.day

dfr = dfr.drop('DATE', axis=1)
dfr = dfr.drop('index', axis=1)


X = dfr.drop('TAVG', axis=1)  
Y = dfr['TAVG'] 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Predict on test data
Y_pred = rf.predict(X_test)

Y_test_indexreset = Y_test.reset_index(drop=True)

plt.plot(Y_test_indexreset,label='Test Data')
plt.plot(Y_pred, color='red')
plt.legend()

mape4 = mean_absolute_percentage_error(Y_test, Y_pred)
mape4
