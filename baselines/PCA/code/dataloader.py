import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import seaborn
import pandas as pd
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import recall_score
import re

import time
import collections
import os

class LogReader:
    def __init__(self, log_format, log_name, indir='./', outdir='./result/', rex=[], every_n=10, max_lines=2000000):
        self.path = indir
        self.logName = log_name
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.every_n = every_n
        self.max_lines = max_lines
    def log_to_dataframe(self, log_file, regex, headers, logformat):
            """ Function to transform log file to dataframe 
            """
            log_messages = []
            linecount = 0
            
            if self.max_lines:
                with open(log_file, 'r', encoding="latin-1") as fin:
                    for i,  line in enumerate(fin):
                        if i % self.every_n == 0:
                            try:
                                match = regex.search(line.strip())
                                message = [match.group(header) for header in headers]
                                log_messages.append(message)
                                linecount += 1
                            except Exception as e:
                                pass
                        if i==self.max_lines:
                            break
            else:
                with open(log_file, 'r', encoding="latin-1") as fin:
                    for i,  line in enumerate(fin):
                        if i % self.every_n == 0:
                            try:
                                match = regex.search(line.strip())
                                message = [match.group(header) for header in headers]
                                log_messages.append(message)
                                linecount += 1
                            except Exception as e:
                                pass
            logdf = pd.DataFrame(log_messages, columns=headers)
            logdf.insert(0, 'LineId', None)
            logdf['LineId'] = [i + 1 for i in range(linecount)]
            return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)
        
    

def get_data(log_file, input_dir, output_dir, log_format, regex=[], every_n=10, aux=0, max_lines=5000000):
    '''Read three dataset sets

    Return:
        df_normal = normal data
        
        df_anomalies = anomaly data
        
        log_payload = normal and anomaly data
        
        true_labels = normal and anomaly labels
        
    '''
    reader = LogReader(log_format, log_file, indir=input_dir, outdir=output_dir, rex=regex, every_n=every_n, max_lines=max_lines)
    reader.load_data()
    log_payload, true_labels = reader.df_log.Content, np.where(reader.df_log.t.values=='-',0,1)
    del reader
    if aux != 0:
        df_anomalies = log_payload.iloc[true_labels.flatten()==1].sample(n=aux).values
        df_normal = log_payload.iloc[true_labels.flatten()==0].sample(n=aux).values
        return df_normal, df_anomalies
    else:
        return log_payload, true_labels
    
    
def get_data_a(log_file, input_dir, output_dir, log_format, regex=[], every_n=10, aux=0, max_lines=2000000):
    '''Read RAS dataset 
    
    Return:
    df_normal = normal data
    
    df_anomalies = anomaly data
    
    log_payload = normal and anomaly data
    
    true_labels = normal and anomaly labels
    '''
    reader = LogReader(log_format, log_file, indir=input_dir, outdir=output_dir, rex=regex, every_n=every_n, max_lines=max_lines)
    reader.load_data()
    log_payload, true_labels = reader.df_log.Content, np.where(reader.df_log.t.values=='FATAL',0,1)
    del reader
    if aux != 0:
        df_anomalies = log_payload.iloc[true_labels.flatten()==1].sample(n=aux).values
        df_normal = log_payload.iloc[true_labels.flatten()==0].sample(n=aux).values
        return df_normal, df_anomalies
    else:
        return log_payload, true_labels


def read_auxiliary_data():

    '''Read auxiliary dataset
    
       Return: 
       
        aux_anomalies0 = auxiliary data
    '''
    log_file = 'Intrepid_RAS_0901_0908_scrubbed'
    input_dir  = '../data_small'
    output_dir = '../outputs/' 
    log_format = '<f> <a>          <c>       <d>                  <e>    <t> <Content>'  #RAS
    regex = []
    every_n = 1
    aux_anomalies_t, tl = get_data_a(log_file, input_dir, output_dir, log_format, regex, 
                                    every_n=every_n, aux=False, max_lines=False)
    aux_anomalies0 = aux_anomalies_t[tl==0]

    return aux_anomalies0
 


def read_target_data(aux_anomalies0,targetfile):
    
    '''Read both normal and anomaly data from target dataset based on the 
        targetfile = 'bgl' or 'spirit' or 'thunderbird'
        
       Read ONLY anomaly data from the unselected two datasets as auxiliary data
       
       
       Return: 
       
        concat_data = concatenated data from target datasets and three auxiliary data
        
        concat_labels = concatenated data labels from target datasets and three auxiliary data
        
        df_size = the size of target dataset
    
    '''

    # Specify the number of Anomaly dataset will be extracted if three data are considered as auxiliary dataset 
    aux_tbird = 226287
    aux_spirit = 764890
    aux_bgl  = 348460



    # Specify the log template for each target dataset

    log_format_tbird = '<t> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
    log_format_spirit = '<t> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>'
    log_format_bgl = '<t> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'

    # specify the data location and output location
    input_dir  = '../data_small/'  
    output_dir = '../outputs/'  
    
    # Initialize the Regular expression list 
    regex = [] 
    # specify whether skip reading the data, 1 = do not skip, read all the data
    every_n = 1
    # Read data
    if targetfile == 'bgl': 
        target_filename = 'bgl2'
        log_file = 'bgl2.log'
        log_format = log_format_bgl
        # aux = 0 means read current dataset as target dataset
        aux_size = 0
        log_payload, true_labels = get_data(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)
        # read the other two datasets as auxiliary dataset
        #1 
        log_file = 'spirit2_small_5m'
        log_format = log_format_spirit
        aux_size = aux_spirit
        _,aux_anomalies1 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
        log_file = 'tbird2_small_5m'
        log_format = log_format_tbird
        aux_size = aux_tbird
        _,aux_anomalies2 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
        
    elif targetfile == 'spirit':
        #read the target dataset
        log_file = 'spirit2_small_5m'
        log_format = log_format_spirit
        # aux = 0 means read current dataset as target dataset
        aux_size = 0
        log_payload, true_labels = get_data(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)
        # read the other two datasets as auxiliary dataset
        #1 
        log_file = 'bgl2.log'
        log_format = log_format_bgl  
        aux_size = aux_bgl
        _,aux_anomalies1 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
        #2
        log_file = 'tbird2_small_5m'
        log_format = log_format_tbird
        aux_size = aux_tbird
        _,aux_anomalies2 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
    elif targetfile == 'thunderbird':
        #read the target dataset
        log_file = 'tbird2_small_5m'
        log_format = log_format_tbird 
        # aux = 0 means read current dataset as target dataset
        aux_size = 0
        log_payload, true_labels = get_data(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)
        # read the other two datasets as auxiliary dataset
        #1 
        log_file = 'bgl2.log'
        log_format = log_format_bgl
        aux_size = aux_bgl
        _,aux_anomalies1 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
        #2 
        log_file = 'spirit2_small_5m'
        log_format = log_format_spirit
        aux_size = aux_spirit
        _,aux_anomalies2 = get_data(log_file, input_dir, output_dir, log_format, regex, 
                                            every_n=every_n, aux=aux_size, max_lines=False)
    # concatenate the two auxiliary datasets to the RAS auxiliary dataset
    concat_anomaly = np.append(aux_anomalies1, aux_anomalies2) 
    concat_anomaly = np.append(concat_anomaly, aux_anomalies0)
    # concatenate target dataset with other three auxiliary dataset
    concat_data = np.append(log_payload.values.reshape(-1,1), concat_anomaly.reshape(-1,1), axis=0)
    true_labels = true_labels.reshape(-1,1)# reshape  
    concat_labels = np.append(true_labels,np.ones(len(concat_anomaly)).reshape(-1,1), axis=0).flatten()
    df_size = len(log_payload)
    
    
    return concat_data,concat_labels,df_size



def train_test_split(data,labels,ratio,df_size):
    '''Split train and test set

       Return: 
       
        x_train = training data
        
        y_train = training labels
        
        x_test = test data
        
        y_test = test data


    '''
    train_size = round(df_size * ratio)
    print(f'train_size = {train_size}')
    '''Split train set'''
    x_train = np.append(data[:train_size][labels[:train_size]==0], 
                                        data[df_size:],axis=0)
    y_train = np.append(labels[:train_size][labels[:train_size]==0].flatten(), 
                                    labels[df_size:].flatten(),axis=0)
    '''Split test set'''    
    
    '''shuffle data'''
    idx = np.random.permutation(len(x_train))
    x_train,y_train = x_train[idx], y_train[idx]
    x_test = data[train_size:df_size]
    y_test = labels[train_size:df_size]
    idx = np.random.permutation(len(x_test))
    x_test,y_test = x_test[idx], y_test[idx]
    print(f"train label = {collections.Counter(y_train)}; test label = {collections.Counter(y_test)}")
    return x_train,y_train,x_test,y_test