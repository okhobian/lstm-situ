import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import minmax_scale
# from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
# from keras_preprocessing import sequence

import warnings


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
            
    def train_data(self, window_size, label_ahead):
        trainX = []
        trainY = []
        
        _df = self.df.iloc[0:, :-1].reset_index(drop=True)    # remove header row & timestamp col
        _X = self.df.iloc[0:, :-2].reset_index(drop=True)      # remove header row & activity+timestamp col
        _Y = pd.get_dummies(self.df['Activity'])
       
        if _Y.shape[1] != len(self.activities):
            warnings.warn("[SELF-WARNING]: dataset contains less type of activities.")
            
        # print(_X.head(10))
        # print("_X >> ", _X.shape)
        # print(_Y.head(10))
        # print("_Y >> ", _Y.shape)
        
        # sliding window to compose multi-var train set
        # for i in range(window_size, len(_df) - label_ahead +1):
        #     trainX.append(_df.iloc[i - window_size:i, 0:29].values.tolist())
        #     trainY.append(_df.iloc[i + label_ahead - 1:i + label_ahead, 0].values.tolist())
        for i in range(window_size, len(_df) - label_ahead +1):
            trainX.append(_X.iloc[i - window_size:i, ].values.tolist())
            # trainY.append(_Y.iloc[i + label_ahead - 1:i + label_ahead].values.tolist()) # (18817, 1, 4)
            trainY.append(_Y.iloc[i + label_ahead - 1:i + label_ahead].values.reshape(-1,).tolist()) # (18817, 4)
        
        trainX, trainY = np.array(trainX), np.array(trainY)
                
        print('trainX shape == {}.'.format(trainX.shape))
        print('trainY shape == {}.'.format(trainY.shape))
        
        return trainX, trainY