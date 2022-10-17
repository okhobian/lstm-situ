from data import *
from models import *
from history_callback import *
from matplotlib import pyplot as plt

## VARIABLES
BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset.csv'
ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']
WINDOW_SIZE = 15
LABEL_AHEAD = 1

## LOAD DATA
data = DATASET()
data.load_data(DATA_FILE, ACTIVITIES)
trainX, trainY = data.train_data(WINDOW_SIZE, LABEL_AHEAD)

## BUILD MODEL
model = build_lstm(trainX.shape[1], trainX.shape[2], trainY.shape[1])

## TRAIN MODEL
histories = Histories()
model.fit(trainX, trainY, batch_size=20, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),

## RESULTS
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
plt.figure()

plt.plot(histories.times)
plt.title('training time')
plt.ylabel('time')
plt.xlabel('batch')
plt.legend(['time'], loc='upper left')

plt.show()