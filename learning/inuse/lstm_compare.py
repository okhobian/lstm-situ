from data import *
from models import *
from history_callback import *
from plot import *

## VARIABLES
BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset_10_01_ignore_ts.csv'
# ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']
ACTIVITIES = ['sleep', 'getup', 'eat', 'work', 'leisure', 'watchtv', 'rest', 'cook', 'goout', 'other']
WINDOW_SIZE = [5, 10, 15, 20]
LABEL_AHEAD = 1

## LOAD DATA
data = DATASET()
data.load_data(DATA_FILE, ACTIVITIES)
histories = Histories()
plt = PLOT()

accuracies = []
losses = []
times = []
for size in WINDOW_SIZE:
    trainX, trainY = data.train_data(size, LABEL_AHEAD)
    model = build_lstm(trainX.shape[1], trainX.shape[2], trainY.shape[1])
    model.fit(trainX, trainY, batch_size=50, epochs=1, verbose=1, callbacks=[histories])  # validation_data=(x_test, y_test),
    accuracies.append(histories.accuracies)
    losses.append(histories.losses)
    times.append(histories.times)


plt.add_multi_data_figure(accuracies, 'Accuracies', 'batch#', 'accuracy', WINDOW_SIZE)
plt.add_multi_data_figure(losses, 'Losses', 'batch#', 'loss', WINDOW_SIZE)
plt.add_multi_data_figure(times, 'Training time', 'batch#', 'time', WINDOW_SIZE)
plt.show_all()