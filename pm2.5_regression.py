import datetime as dt
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
# Functions that help me get data
import helper

# Wind speed data
wind_data = []
for month in [5, 6, 7, 8]:
    if month == 5:
        r = range(11, 32)
    elif month == 6:
        r = range(1, 31)
    elif month == 7:
        r = range(1, 32)
    elif month == 8:
        r = range(1, 28)
    for day in r:
        tmp = helper.crawler(month, day)
        wind_data = wind_data + tmp
        
wind = pd.DataFrame(wind_data)
wind.set_index(['month', 'day', 'hour'], inplace=True)

# Rainfall data
rain_data = []
for month in [5, 6, 7, 8]:
    if month == 5:
        r = range(11, 32)
    elif month == 6:
        r = range(1, 31)
    elif month == 7:
        r = range(1, 32)
    elif month == 8:
        r = range(1, 28)
    for day in r:
        tmp = helper.crawler_rain(month, day)
        rain_data = rain_data + tmp
        
rain = pd.DataFrame(rain_data)
rain.set_index(['month', 'day', 'hour'], inplace=True)

# Load data
n = 1
pos = 5
data = helper.get_data_by_pos(pos)
df = pd.DataFrame(data)
# Input time
if pos == 2:
    time = ['2019 06 06', '2020 01 01']
else:
    time = ['2019 05 11', '2020 01 01']
taipei_tz = pytz.timezone('Asia/Taipei')
# Set time
start_time = dt.datetime.strptime(time[0], '%Y %m %d').replace(tzinfo=taipei_tz)
end_time = dt.datetime.strptime(time[1], '%Y %m %d').replace(tzinfo=taipei_tz)
# Select the duration
df = df.loc[ df['date'] >= start_time ]
df = df.loc[ df['date'] <= end_time ]
# Rename the names of columns
df = df.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
# Data cleaning
df = df.loc[ df['pm2.5'] <= 120 ]
df = df.loc[ df['humidity'] <= 100 ]
# Split time infomation from column `date`
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)
df['hour'] = df['date'].apply(lambda x: x.hour)
# Discard some columns
df = df.drop(columns=['date'])
# Evaluate mean values for each hour
dfmean = df.groupby(['month', 'day', 'hour']).mean()
# Concat !!!!!!!!
dfconcat = pd.concat([dfmean, wind, rain], axis=1, sort=False)
# Reset index
dfconcat.reset_index(inplace=True)
# Reconstruct time infomation by `month`, `day`, and `hour`

def get_time(x):
    time_str = '2019 %d %d %d' % (x[0], x[1], x[2])
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = dt.datetime.strptime(time_str, '%Y %m %d %H').replace(tzinfo=taipei_tz)
    return time

dfconcat['time'] = dfconcat[['month', 'day', 'hour']].apply(get_time, axis=1)
# Shift columns
dfconcat[['pm2.5_shift']] = dfconcat[['pm2.5']].shift(-n)
dfconcat[['time_shift']] = dfconcat[['time']].shift(-n)
dfconcat[['pm2.5_p1']] = dfconcat[['pm2.5']].shift(1)
dfconcat[['pm2.5_p2']] = dfconcat[['pm2.5']].shift(2)
dfconcat[['pm2.5_p3']] = dfconcat[['pm2.5']].shift(3)
dfconcat[['pm2.5_p4']] = dfconcat[['pm2.5']].shift(4)
dfconcat[['pm2.5_p5']] = dfconcat[['pm2.5']].shift(5)
dfconcat[['temp_p1']] = dfconcat[['temp']].shift(1)
dfconcat[['temp_p2']] = dfconcat[['temp']].shift(2)
dfconcat[['temp_p3']] = dfconcat[['temp']].shift(3)
dfconcat[['temp_p4']] = dfconcat[['temp']].shift(4)
dfconcat[['temp_p5']] = dfconcat[['temp']].shift(5)
dfconcat[['humidity_p1']] = dfconcat[['humidity']].shift(1)
dfconcat[['humidity_p2']] = dfconcat[['humidity']].shift(2)
dfconcat[['humidity_p3']] = dfconcat[['humidity']].shift(3)
dfconcat[['humidity_p4']] = dfconcat[['humidity']].shift(4)
dfconcat[['humidity_p5']] = dfconcat[['humidity']].shift(5)
dfconcat[['speed_p1']] = dfconcat[['speed']].shift(1)
dfconcat[['speed_p2']] = dfconcat[['speed']].shift(2)
dfconcat[['speed_p3']] = dfconcat[['speed']].shift(3)
dfconcat[['speed_p4']] = dfconcat[['speed']].shift(4)
dfconcat[['speed_p5']] = dfconcat[['speed']].shift(5)
dfconcat[['rain_p1']] = dfconcat[['rain']].shift(1)
dfconcat[['rain_p2']] = dfconcat[['rain']].shift(2)
dfconcat[['rain_p3']] = dfconcat[['rain']].shift(3)
dfconcat[['rain_p4']] = dfconcat[['rain']].shift(4)
dfconcat[['rain_p5']] = dfconcat[['rain']].shift(5)
# Choose data every 6 row
# dfconcat = dfconcat.loc[ (dfconcat.index % 6) == 0 ]
# Discard rows that contain NaN value
dfconcat.dropna(inplace=True)
# Save mean and std
feature_cols = ['pm2.5', 'humidity', 'speed', 'rain', 'pm2.5_p1'] 
label_cols = ['pm2.5_shift']
want_cols = feature_cols + label_cols

mean_all = dfconcat.loc[:, want_cols].mean()
std_all = dfconcat.loc[:, want_cols].std()
# Normalization
dfconcat.loc[:, want_cols] = (dfconcat.loc[:, want_cols] - mean_all) / std_all
# Divid training set and test set
train_size = len(dfconcat)*0.8
train_size = int(train_size)

train_df = dfconcat[:train_size]
test_df = dfconcat[train_size:]
# Select features
train_X = train_df[feature_cols]
train_y = train_df[label_cols]

test_X = test_df[feature_cols]
test_y = test_df[label_cols]
# Fit the model
model = linear_model.LinearRegression(normalize=True)
model.fit(train_X, train_y)
# See the coefficients of our model
for i in range(len(train_X.columns)):
    print('Coefficient for %10s:\t%s' % (train_X.columns[i], model.coef_[0][i]))
print('Intercept: \t\t\t %s' % model.intercept_[0])
# Calculate predict value
predict_train_y = model.predict(train_X)
predict_test_y = model.predict(test_X)
# Transform normalized data back to original data
m = mean_all['pm2.5_shift']
s = std_all['pm2.5_shift']

test_y_ori = test_y * s + m
predict_test_y_ori = predict_test_y * s + m

train_y_ori = train_y * s + m
predict_train_y_ori = predict_train_y * s + m
# Calculate MSE, MAPE for training set & test set

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_mse = metrics.mean_squared_error(train_y_ori, predict_train_y_ori)
test_mse = metrics.mean_squared_error(test_y_ori, predict_test_y_ori)

train_mape = mean_absolute_percentage_error(train_y_ori, predict_train_y_ori)
test_mape = mean_absolute_percentage_error(test_y_ori, predict_test_y_ori)

print('Train MSE:\t %f,\t RMSE: %f (μg/m^3),\t MAPE:\t %f %%' % (train_mse, np.sqrt(train_mse), train_mape))
print('Test MSE:\t %f,\t RMSE: %f (μg/m^3),\t MAPE:\t %f %%' % (test_mse, np.sqrt(test_mse), test_mape))
# Add explicitly converter
pd.plotting.register_matplotlib_converters()
# Plt
plt.figure(figsize=(12, 7))
plt.plot(test_df['time_shift'], test_y_ori, label='actual values')
plt.plot(test_df['time_shift'], predict_test_y_ori, label='predict values')
plt.xticks(rotation=45)
plt.ylabel('Time')
plt.ylabel('PM2.5 (μg/m^3)')
plt.legend(loc='upper right')
plt.title('Linear Regression: Testing Set - Position %d' % pos)
plt.grid()
plt.tight_layout()
plt.show()
