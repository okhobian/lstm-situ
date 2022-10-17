from re import X
from data import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
DATA_FILE = f'{BASE_PATH}/datasets/adlnormal/data'

columns = ["Date", "Time", "Sensor", "Sensor_Status", "Activity", "Activity_Status"]
activities = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']

data = ADLNORMAL()
data.load_data(DATA_FILE, activities, columns)
feature, label, X_train, X_test, y_train, y_test = data.compose_train_test_sets()

model = Sequential()
model.add(Embedding(30, 30, input_length=216))
model.add(LSTM(50))
model.add(Dense(5, activation='softmax'))
print(model.summary())

# print(X_train.shape)
print(y_train.shape)

# print(X_train[0])
# print(y_train[0])

## Train ########################
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=5)

## Test ########################
# The evaluate() method - gets the loss statistics
print("======== eval ==========")
model.evaluate(X_test, y_test, batch_size=1)     

## Predict ########################
# The predict() method - predict the outputs for the given inputs
print("======== pred ==========")
seq = np.array([[1],[2],[9],[10],[3],[2],[1],[9],[10],[3],[4],[4],[4],[3],[3],[4],[3],[13],[3],[3],[13]]) # 01000 -> wash
test_sample = np.array(sequence.pad_sequences(seq, maxlen=216, value=0))

predicted = (model.predict(test_sample) > 0.5).astype("int32")
# predicted = model.predict(test_sample) 
print(predicted[0])

print(feature)
print(label)
    



## Plot ########################
# history_dict = history.history
# print(history_dict.keys())

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()
plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# import matplotlib.pyplot as plt

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'b.', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'b.', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()
