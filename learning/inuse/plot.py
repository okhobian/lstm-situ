from matplotlib import pyplot as plt

class PLOT():
    def __init__(self):
        self.plt = plt
        
    def add_figure(self, data, title, xlabel, ylabel, legend):
        self.plt.plot(data)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.legend(legend, loc='upper left')
        self.plt.figure()

    def show_all(self):
        self.plt.show()

# plt.plot(histories.losses)
# plt.title('model losses')
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['loss'], loc='upper left')
# plt.figure()

# plt.plot(histories.accuracies)
# plt.title('model accuracies')
# plt.ylabel('accuracy')
# plt.xlabel('batch')
# plt.legend(['accuracy'], loc='upper left')
# plt.figure()

# plt.plot(histories.times)
# plt.title('training time')
# plt.ylabel('time')
# plt.xlabel('batch')
# plt.legend(['time'], loc='upper left')

# plt.show()