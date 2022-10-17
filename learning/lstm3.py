from re import X
from data2 import *
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset.csv'

activities = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']

data = DATASET()
data.load_data(DATA_FILE, activities)


trainX, trainY = data.compose_train_test_sets(15, 1)


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1], activation='softmax'))
# model.add(Dense(trainY.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs=5, batch_size=1, validation_split=0.3, verbose=1)


plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_accuracy', 'val_loss'], loc='upper left')
# plt.figure()


# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
plt.show()