import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale


def load_data(filename, columns):
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    df.columns = columns
    # print(df.shape)
    return df

def one_hot_mapping(dataframe, columns):
    
    '''
        columns: columns that need to be encoded
        ["Sensor"]
    '''
    
    # compute encoding for one-hot columns
    col_mappings = {}
    for col in columns:
        
        # get unique values from the column (with nan values ignored)
        unique_col_vals = dataframe[col].dropna().unique()
        values = array(unique_col_vals)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # print(integer_encoded)
        
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # print(onehot_encoded)
    
        ## invert first example
        # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
        # print(">>>", inverted)
        
        
        '''
            "column_name": {
                "sensor_name": "sensor_one_hot_encode"
            }
        '''
        col_mappings[col] = dict(zip(unique_col_vals, onehot_encoded))
        
    return col_mappings    

def vectorize_dataset(dataframe, mappings):
    
    '''
        subject to change, very specific to this dataset
    '''
    
    on_definitions = ['ON', 'START', 'OPEN', 'PRESENT', 'START_INSTRUCT']
    off_definitions = ['OFF', 'END', 'CLOSE', 'ABSENT', 'STOP_INSTRUCT']
    
    # define new columns in vectorized dataframe (all sensors + sensor status)
    _new_columns = ['']* (len(mappings['Sensor'].keys())+1)
    for k, v in mappings['Sensor'].items():
        _new_columns[argmax(v)] = k # order sensor names based on one-hot encoding positions
        
    _new_columns[-1] = 'Sensor_Status'
    vectorized_data = pd.DataFrame(columns=_new_columns)
    # print(_new_columns)

    # go over raw data by row
    for index, row in dataframe.iterrows():
        
        # if index > 0: break
        
        # new encoded row
        sensor_one_hot_encode = list(mappings['Sensor'][row['Sensor']])
        
        # deal with sensor status
        sensor_status = 'NaN'
        if row['Sensor_Status'] in on_definitions:
            sensor_status = 1
        elif row['Sensor_Status'] in off_definitions:
            sensor_status = 0
        else:
            sensor_status = row['Sensor_Status']
        
        # append sensor status to the end of each row
        sensor_one_hot_encode.extend([sensor_status])

        # new row as dict to be added to the dataframe
        _new_row = dict(zip(_new_columns, sensor_one_hot_encode))
        vectorized_data = pd.concat([vectorized_data, pd.DataFrame([_new_row])])
    
    # scale sensor values to between 0 and 1; may need further implementation for ON_OFF situations
    vectorized_data[['Sensor_Status']] = minmax_scale(vectorized_data[['Sensor_Status']])
    
    return vectorized_data

def extract_activities(dataframe, activity_name):
    activity = dataframe.loc[dataframe['Activity'] == activity_name]
    activity_begins = activity.iloc[::2, :].index.values        # idx of every 2nd row starting from the first row
    activity_ends = activity.iloc[1::2, :].index.values         # idx of every 2nd row starting from the second row
    activity_idx = list(zip(activity_begins, activity_ends))    # list of (begin_idx, end_idx)

    # group each phone call activity into a list
    all_activities = []
    for idx in activity_idx:
        all_activities.append(dataframe.loc[idx[0]:idx[1]])
        
    return all_activities

def basic_statics(data:list):
    return np.mean(data), np.std(data), np.var(data)

# def preprocessing_batches(folder):
    


if __name__ == '__main__':
    
    BASE_PATH = '/Users/hobian/Desktop/GitHub/lstm-situ'
    
    sensor_type = {
        "M": "motion sensor",
        "I01": "oatmeal sensor",
        "I02": "raisins sensor",
        "I03": "brown sugar sensor",
        "I04": "bowl sensor",
        "I05": "measuring spoon sensor",
        "I07": "pot sensor",
        "I08": "phone book sensor",
        "D01": "cabinet sensor",
        "AD1-A": "water sensor",
        "AD1-B": "water sensor",
        "AD1-C": "burner sensor",
        "asterisk": "phone usage"
    }

    columns = ["Date", "Time", "Sensor", "Sensor_Status", "Activity", "Activity_Status"]
    activities = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']
    
    data = load_data(f'{BASE_PATH}/datasets/adlnormal/data', columns)
    mapping = one_hot_mapping(data, ["Sensor", "Activity"])
    output_mapping = one_hot_mapping(pd.DataFrame({"Activities": activities}), ["Activities"])

    all_activities = {} # {'activity_name':[first_group_of_rows, second_group_of_rows, ...]}
    for i, activity in enumerate( activities ): # each activity
        
        # if i > 0: break
        
        all_activities[activity] = extract_activities(data, activity)
        
        batch_sizes = []
        batches = []
        for j, a in enumerate( all_activities[activity] ):  # each batch of the same activity
            
            batch_sizes.append(len(a))
            vectorized_data = vectorize_dataset(a, mapping) # dataframe
            batches.append(vectorized_data.to_numpy())      # list of numpy arrays
            print(vectorized_data)
            # vectorized_data = vectorized_data.to_numpy()
            batch_vectorized_label = output_mapping['Activities'][activity]
            
            # print(vectorized_data.shape)
            # print(batch_vectorized_label.shape)
            # batch_data = np.stack((vectorized_data, batch_vectorized_label), axis=2)
            # # print(batch_data)
            # print(batch_data.shape)
            # print("--------------------------------")
            
            # # output onehot batches to csv files
            # out_file = f'{BASE_PATH}/datasets/onehot/{activity}/{activity}_onehot_batch_{j}.csv'
            # vectorized_data.to_csv(out_file, sep='\t')
        
        
        # print(batches)
        
        
        # # compute dataset statics
        # batch_sizes.sort()
        # mean, std, var = basic_statics(batch_sizes)
        # summary_file = f'{BASE_PATH}/datasets/onehot/summary.txt'
        # summary = f'{activity}:\t mean:{mean}, \t std:{std}, \t var:{var} \n batch_sizes: {batch_sizes} \n\n'
        # with open(summary_file, "a") as myfile:
        #     # myfile.seek(0)                       
        #     # myfile.truncate()
        #     myfile.write(summary)
                         

    # example of converting second group of rows of the activity into one_hot_dataframe
    # activity_series = all_activities['Cook'][1]
    # print(activity_series)
    # print("======================")
    # vectorized_data = vectorize_dataset(activity_series, mapping)
    # print(vectorized_data)
 