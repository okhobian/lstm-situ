import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale


class ADLNORMAL:
    def __init__(self, ):
        self.columns = None
        self.activities = None
        self.df = None
        
    def load_data(self, filename, columns, activities):
        self.columns = columns
        self.activities = activities
        self.df = pd.read_csv(filename, delim_whitespace=True, header=None)
        self.df.columns = self.columns

    def extract_sequences(self, activity_name):
                
        '''
        activity = self.df.loc[self.df['Activity'] == activity_name]
        activity_begins = activity.iloc[::2, :].index.values        # idx of every 2nd row starting from the first row
        activity_ends = activity.iloc[1::2, :].index.values         # idx of every 2nd row starting from the second row
        activity_idx = list(zip(activity_begins, activity_ends))    # list of (begin_idx, end_idx)

        ## list of (sensor_sequence, activity)
        sequence_label = []
        for i, idx in enumerate( activity_idx ):
            if i>3: break 
            sequence = self.df.loc[idx[0]:idx[1]]
            sensor_sequence = sequence.loc[:,'Sensor'].to_numpy()
            sequence_label.append((sensor_sequence, activity_name))
            
        features, labels = zip(*sequence_label)
            
        ## [sensor_sequence] [activity]
        # return np.array(features), np.array(labels)
        '''
        
        sequences = []
        labels = []
        for curr_activity in self.activities:
            
            ## extract all begin_end indices for the current activity
            activity = self.df.loc[self.df['Activity'] == curr_activity]
            activity_begins = activity.iloc[::2, :].index.values        # idx of every 2nd row starting from the first row
            activity_ends = activity.iloc[1::2, :].index.values         # idx of every 2nd row starting from the second row
            activity_idx = list(zip(activity_begins, activity_ends))    # list of (begin_idx, end_idx)
            
            ## for all instances of the current activity
            for i, idx in enumerate( activity_idx ):
                if i>3: break 
                sequence = self.df.loc[idx[0]:idx[1]]
                sensor_sequence = sequence.loc[:,'Sensor'].to_numpy()
                sequences.append(np.array(sensor_sequence))
                labels.append(np.array(curr_activity))
            
        # sequences = np.ravel(np.array(sequences))   # 1D list of all sequences
        # labels = np.ravel(np.array(labels))         # 1D list of all labels
        
        return np.array(sequences), np.array(labels)
    
    def compose_train_test_sets(self):
        s, l = self.extract_sequences(self.activities)
        print(l)
        print(type(l))
        print(l.shape)
        
        # sequences = []
        # labels = []
        # for a in activities:
        #     s, l = self.extract_sequences(a)
        #     sequences.append(s)
        #     labels.append(l)
        
        # sequences = np.ravel(np.array(sequences))   # 1D list of all sequences
        # labels = np.ravel(np.array(labels))         # 1D list of all labels
        
        return None, None

if __name__ == '__main__':
    
    BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
    DATA_FILE = f'{BASE_PATH}/datasets/adlnormal/data'

    columns = ["Date", "Time", "Sensor", "Sensor_Status", "Activity", "Activity_Status"]
    activities = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']
    
    data = ADLNORMAL()
    data.load_data(DATA_FILE, columns, activities)
    train, test = data.compose_train_test_sets()
    
    
    
    
    # mapping = one_hot_mapping(data, ["Sensor", "Activity"])
    # output_mapping = one_hot_mapping(pd.DataFrame({"Activities": activities}), ["Activities"])