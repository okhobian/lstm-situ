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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# history = model.fit(trainX, trainY, epochs=5, batch_size=1, validation_split=0.3, verbose=1)

###################################################

from keras.callbacks import Callback

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))


histories = Histories()

model.fit(trainX, trainY, batch_size=20, epochs=1, verbose=1,
        #   validation_data=(x_test, y_test),
          callbacks=[histories]
         )


plt.plot(histories.losses)
plt.title('model losses')
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['loss'], loc='upper left')
plt.figure()

plt.plot(histories.accuracies)
plt.title('model accuracies')
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.legend(['accuracy'], loc='upper left')

# plt.plot(histories.accuracies)
# plt.plot(histories.losses)
# plt.title('model accuracies & losses')
# plt.ylabel('acc')
# plt.xlabel('batch')
# plt.legend(['acc', 'loss'], loc='upper left')


plt.show()