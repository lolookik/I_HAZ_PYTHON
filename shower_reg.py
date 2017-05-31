import matplotlib
matplotlib.use('Qt4Agg')

#import data
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from math import sqrt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#load dataset
def fun_date_parser(x):
	return datetime.strptime('190'+x, '%Y-%m')


#frame a sequence as a supervied learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	col = [df.shift(i) for i in range(1, lag+1)]
	col.append(df)
	df = concat(col, axis=1)
	df.fillna(0, inplace=True)
	return df


#difference series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


#invert difference value
def inverse_difference(history, yhat, interval=1):
	return  yhat + history[-interval]


# scale train and test to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    #transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    #transform test
    test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


#inverse scaling for forcasting
def inverse_scale(scaler, X, val):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inver = scaler.inverse_transform(array)
    return inver[0, -1]


#fit LSTM on train data
def fit_lstm(train, batch_S, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_S, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs = 1, batch_size = batch_S, verbose=0, shuffle=False)
        model.reset_states()
    return model
    
    
#one step forecast
def forecast_lstm(model, batch_S, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_S)
    return yhat[0,0]
                 

#####################################################################
#BEGIN PROG
#####################################################################
#load data
series = read_csv('DEXUSEU.csv', header=0, parse_dates=False, index_col=0, squeeze=True)#, date_parser=fun_date_parser)

#stationarity transformation
raw_values = series.values
diff_values = difference(raw_values, 1)

#slide values for supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
super_values = supervised.values

#split data train-test 
split_val = 50
train, test = super_values[0:-split_val], super_values[-split_val:]

#scale data
scaler, train_scaled, test_scaled = scale(train, test)

#fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 3)

#forecasting training dataset
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

#validation
predictions = list()
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    yhat = inverse_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    predictions.append(yhat)
    expected = raw_values[len(train)+i+1]
    print('M = %d   Pred = %f   Exp = %f' % (i+1, yhat, expected))
    
#evaluation
epsi = sqrt(mean_squared_error(raw_values[-split_val:], predictions))
print('epsi: %.3f' % epsi)
pyplot.plot(raw_values[-split_val:])
pyplot.plot(predictions)
pyplot.show()

z_vec = numpy.ones(raw_values.shape[0]-split_val)*numpy.mean(raw_values[0])
pred_vec = numpy.hstack((z_vec, predictions))
pyplot.plot(raw_values)
pyplot.plot(pred_vec)
pyplot.show()


print("endscript")
