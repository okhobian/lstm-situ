import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import minmax_scale
# from sklearn.model_selection import train_test_split

# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
# from keras_preprocessing import sequence

class DATASET:
    def __init__(self, ):
        self.columns = None
        self.activities = None
        self.df = None
    
    def load_data(self, filename, activities, columns=None):
        if columns:
            self.df = pd.read_csv(filename, delim_whitespace=True, header=None)
            self.columns = columns
            self.activities = activities
            self.df.columns = self.columns
        else:
            self.df = pd.read_csv(filename)
            self.columns = list(self.df.columns)
            self.activities = activities
            self.df.columns = self.columns
            # print(self.columns)
            # print(len(self.columns))
            # print(self.activities)
            # print(self.df.shape[0])
            # print(self.df)
            print(self.df.shape)
            
    def compose_train_test_sets(self, window_size, label_ahead):
        trainX = []
        trainY = []
        
        _df = self.df.iloc[1:, :-1]     # remove header row & timestamp col
        
        # sliding window
        for i in range(window_size, len(_df) - label_ahead +1):
            trainX.append(_df.iloc[i - window_size:i, 0:29].values.tolist())
            trainY.append(_df.iloc[i + label_ahead - 1:i + label_ahead, 0].values.tolist())
        
        # _X = self.df.iloc[1:,:-2]       # remove header row & last two columns (activity, timestamp)
        # _Y = self.df['Activity']        # select only activity column as labels
        
        # print('_X shape == {}.'.format(_X.shape))
        # print('_Y shape == {}.'.format(_Y.shape))
                
        # for i in range(window_size, len(_X)):
        #     trainX.append( _X[ i-window_size : i, :] )
        #     trainY.append( _Y[ i : i+1, ] )
        # print(trainX)
        trainX, trainY = np.array(trainX), np.array(trainY)
        
        # print(_X.head(10))
        # print(_X.shape)
        # print('-------------')
        # print(_Y.head(10))
        # print(_Y.shape)
        
        print('trainX shape == {}.'.format(trainX.shape))
        print('trainY shape == {}.'.format(trainY.shape))
        
        return trainX, trainY