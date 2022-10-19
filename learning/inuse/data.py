import copy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import minmax_scale
# from sklearn.model_selection import train_test_split

# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
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
    
    def statics(self):
        fields = {
            "total_sequence": 0,
            "seq_lengths": [],
            "avg_seq_len": 0,
            "max_seq_len": 0,
            "min_seq_len": 0,
            "seq_len_std": 0
        }
        statics = {activity : copy.deepcopy(fields) for activity in self.activities}
        grouped = self.extract_sequences ()
        for _, group in grouped:    # for every activity chunk
            curr_activity = set(group['Activity'])  # remove duplicates from Activity col
            if len(curr_activity) != 1: continue    # invalid chunk
            curr_activity = list(curr_activity)[0]  # get activity string
            statics[curr_activity]["seq_lengths"].append(len(group))

        # results: calculate remaining stats, exclude list of all sequences
        results = {activity : copy.deepcopy(fields) for activity in self.activities}
        for activity in self.activities:
            results[activity]["total_sequence"] = len(statics[activity]["seq_lengths"])
            results[activity]["avg_seq_len"] = int(np.mean(statics[activity]["seq_lengths"]))
            results[activity]["seq_len_std"] = round(float(np.std(statics[activity]["seq_lengths"])),2)
            results[activity]["max_seq_len"] = int(np.max(statics[activity]["seq_lengths"]))
            results[activity]["min_seq_len"] = int(np.min(statics[activity]["seq_lengths"]))
            # np.var(data)
        
        return results
    
    def _plot_stats(self, stats):
        # set width of bar
        barWidth = 0.25
        fig = plt.subplots(figsize =(19, 8))
    
        avgs = []
        maxs = []
        mins = []
        # stds = []
        for activity in self.activities:
            # print(activity)
            # print(stats[activity]['avg_seq_len'])
            avgs.append(stats[activity]['avg_seq_len'])
            maxs.append(stats[activity]['max_seq_len'])
            mins.append(stats[activity]['min_seq_len'])
            # stds.append(stats[activity]['seq_len_std'])
            
        # print(avgs)
        # set height of bar
        # IT = [12, 30, 1, 8, 22]
        # ECE = [28, 6, 16, 5, 10]
        # CSE = [29, 3, 24, 25, 17]
        
        # Set position of bar on X axis
        br1 = np.arange(len(avgs))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # br4 = [x + barWidth for x in br3]
        
        # Make the plot
        plt.bar(br1, avgs, color ='g', width = barWidth,
                edgecolor ='grey', label ='avgs')
        plt.bar(br2, maxs, color ='y', width = barWidth,
                edgecolor ='grey', label ='maxs')
        plt.bar(br3, mins, color ='b', width = barWidth,
                edgecolor ='grey', label ='mins')
        # plt.bar(br4, stds, color ='r', width = barWidth,
        #         edgecolor ='grey', label ='stds')
        
        # Adding Xticks
        plt.xlabel('Activity', fontweight ='bold', fontsize = 15)
        plt.ylabel('Sequence Length', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(self.activities))], self.activities)
        
        for i, v in enumerate(avgs):
            plt.text(br1[i]-0.05, v + 0.2, str(v))
        for i, v in enumerate(maxs):
            plt.text(br2[i]-0.05, v + 0.2, str(v))
        for i, v in enumerate(mins):
            plt.text(br3[i]-0.05, v + 0.2, str(v))
        
        plt.legend()
        plt.show()
    
    def extract_sequences(self):
        grouped = self.df.groupby( (self.df.Activity != self.df.Activity.shift()).cumsum())    # group by each activity       
        # sequences = []
        # labels = []
        # print(grouped)
        # i = 0
        # for _, group in grouped:    # for every activity chunk
        #     sensor_group = group[group.columns.difference(['Activity', 'timestamp'])].to_numpy()    # only sensor values into numpy
        #     sensor_group = [''.join(row.astype(str)) for row in sensor_group]   # join sensor values to binary str
        #     sensor_group = [[int(sensors, 2)] for sensors in sensor_group]      # into [[x1], [x2], [x3], [x4], [x5], [x6]]
            
        #     group = group.reset_index()
        #     # print(group['Activity'][0])
        #     sequences.append(sensor_group)
        #     labels.append(group['Activity'][0])
            # break
            # i+=1
            # if i > 5:
            #     break
        
        # print(sequences)
        # return np.array(sequences), np.array(labels)
        return grouped
    
if __name__ == '__main__':
    ## VARIABLES
    BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
    DATA_FILE = f'{BASE_PATH}/datasets/openshs/dataset_10_01_ignore_ts.csv'
    # ACTIVITIES = ['sleep', 'eat', 'personal', 'work', 'leisure', 'anomaly', 'other']
    ACTIVITIES = ['sleep', 'getup', 'eat', 'work', 'leisure', 'watchtv', 'rest', 'cook', 'goout', 'other']
    
    data = DATASET()
    data.load_data(DATA_FILE, ACTIVITIES)
    s = data.statics()
    print(json.dumps(s, sort_keys=False, indent=4))

    
    data._plot_stats(s)