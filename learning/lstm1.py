from re import X
from data2 import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/adlnormal/data'

columns = ["Date", "Time", "Sensor", "Sensor_Status", "Activity", "Activity_Status"]
activities = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']

data = ADLNORMAL()
data.load_data(DATA_FILE, columns, activities)
X_train, X_test, y_train, y_test = data.compose_train_test_sets()

model = Sequential()
model.add(Embedding(216, 10, input_length=1))
# model.add(Embedding(216, 5, input_length=1))
model.add(LSTM(50))
model.add(Dense(5, activation='softmax'))
print(model.summary())

print(X_train.shape)
print(y_train.shape)

# print(X_train[0])
# print(y_train[0])

## Train ########################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100)

## Test ########################
# The evaluate() method - gets the loss statistics
model.evaluate(X_test, y_test, batch_size=10)     
# returns: loss: 0.0022612824104726315

# The predict() method - predict the outputs for the given inputs
print(X_train[0])
predicted = model.predict(X_train[0]) 
print(predicted[0])
print(predicted.shape)
print(y_train[0])
# returns: [ 0.65680361],[ 0.70067143],[ 0.70482892]



## Plot ########################
# history_dict = history.history
# print(history_dict.keys())

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
loss = history.history['loss']
# val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b.', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b.', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


