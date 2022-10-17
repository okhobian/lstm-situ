from re import X
from data2 import *

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Embedding
# from keras.layers import InputLayer


BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset.csv'

activities = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']

data = DATASET()
data.load_data(DATA_FILE, activities)


trainX, trainY = data.compose_train_test_sets(15, 1)
# print(trainX)
# print(trainY)
# model = Sequential()
# # model.add(Embedding(8929536, 8929536, input_length=maxLen))
# model.add(LSTM(50, return_sequences=True, input_shape=(255,1)))
# model.add(LSTM(50))
# model.add(Dense(7, activation='softmax'))
# model.build((maxLen,))
# print(model.summary())