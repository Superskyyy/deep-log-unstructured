from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def generate(x_train_normal,window_size):
    '''generate train set using dataloder'''
    num_sessions = 0
    inputs = []
    outputs = []
    for line in x_train_normal:
        # print(f'line = {line}')
        num_sessions += 1
        line = tuple(map(lambda n: n - 1, line))
        for i in range(len(line) - window_size):
            inputs.append(line[i:i + window_size])
            outputs.append(line[i + window_size])
    print('Number of sessions: {}'.format(num_sessions))
    print('Number of seqs: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

def generate_test(dataset,window_size):
    '''generate test set '''
    data = []
    for ln in dataset:
        ln = list(map(lambda n: n - 1, ln))
        ln = ln + [-1] * (window_size + 1 - len(ln))
        data.append(tuple(ln))
    print('Number of sessions for test set: {}'.format(len(data)))
    return data

def time_check(msg=''):
    '''print out existing local time'''
    print('{} -- {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))