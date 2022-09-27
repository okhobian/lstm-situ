import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing import sequence



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

    def extract_sequences(self):
               
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
                # if i>0: break 
                sequence = self.df.loc[idx[0]:idx[1]]
                sensor_sequence = sequence.loc[:,'Sensor'].to_numpy()
                sequences.append(np.array(sensor_sequence))
                labels.append(np.array(curr_activity))
            
        # sequences = np.ravel(np.array(sequences))   # 1D list of all sequences
        # labels = np.ravel(np.array(labels))         # 1D list of all labels
        
        ## all activity sequences, all corresponding activities as labels
        return np.array(sequences), np.array(labels)
    
    def compose_train_test_sets(self):
        
        ## 1. onehot s and l;
        s, l = self.extract_sequences()
        
        unique_col_vals = self.df['Sensor'].dropna().unique()   # extract unique sensor types from data
        values = sorted( array(unique_col_vals) )               # sort alphabetically unique sensor types
        word_idx = {k: v+1 for v, k in enumerate(values)}       # reserve 0 for padding
                
        tokenizer = Tokenizer(filters="")                       # empty filter to avoid spliting sensor name with '_'
        tokenizer.word_index = word_idx                         # initialize word index
        
        ## tokenize sequences
        tokenized_seq = []
        for seq in s:
            tokenizer.fit_on_texts(seq)
            sequences = tokenizer.texts_to_sequences(seq) 
            tokenized_seq.append(sequences)
        
        
        # one-hot encode labels
        integer_mapping = {x: i for i,x in enumerate(sorted(set(l)))}
        vec = [integer_mapping[word] for word in l]
        onehot_labels = to_categorical(vec, num_classes=5)
        
        ## example of getting back to real values
        # indices = np.argmax(onehot_labels[0])
        # print(indices)
        # print(l[indices])
        
        # print(len(tokenized_seq))
        # print(len(onehot_labels))
        
        ## 2. padding
        maxLen = len(max(tokenized_seq, key=len))
        padding = sequence.pad_sequences(tokenized_seq, maxlen=maxLen, value=0)
        
        ## 3. spilit into train, test    
        X_train, X_test, y_train, y_test = train_test_split(padding, onehot_labels,
                                                            random_state=42, 
                                                            test_size=0.01, 
                                                            shuffle=True)
            
        return word_idx, integer_mapping, np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        

if __name__ == '__main__':
    
    BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
    DATA_FILE = f'{BASE_PATH}/datasets/adlnormal/data'

    columns = ["Date", "Time", "Sensor", "Sensor_Status", "Activity", "Activity_Status"]
    activities = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']
    
    data = ADLNORMAL()
    data.load_data(DATA_FILE, columns, activities)
    X_train, X_test, y_train, y_test = data.compose_train_test_sets()
    
    
    