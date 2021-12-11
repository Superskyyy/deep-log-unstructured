import sys,time
from preprocess import Preprocessor
from model import Model
from dataloader import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate, generate_test, time_check
import collections 
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader



def run_model(target_filename,window_size):
    '''Start training and evaluate the model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratios = [0.1,0.2,0.4,0.6,0.8]
    for ratio in ratios:
        x_train,y_train,x_test,y_test,label_train,label_test = train_test_split(X,y,labels,ratio)
        x_train = x_train[label_train==0]
        x_train = x_train[:int(0.01*len(x_train))]
        num_candidates = int((num_classes*0.32))
        model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        seq_dataset = generate(x_train,window_size)
        dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        '''Train the model'''
        start_time = time.time()
        total_step = len(dataloader)
        for epoch in range(num_epochs):  # Loop over the dataset multiple times
            train_loss = 0
            for step, (seq, label) in enumerate(dataloader):
                # Forward pass
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
            time_check('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))


        
        '''Evaluate the model'''
        x_test_normal = x_test[label_test == 0]
        x_test_anomaly = x_test[label_test == 1]
        x_test_normal_gen = generate_test(x_test_normal,window_size)
        x_test_anomaly_gen = generate_test(x_test_anomaly,window_size)
        model.eval()
        TP = 0
        FP = 0
        start_time = time.time()
        time_check(f'=== Start evaluating normal data: {len(x_test_normal_gen)} ===')
        with torch.no_grad():
            for j,line in enumerate(x_test_normal_gen):
                if j % 10000 == 0:
                    time_check(f'line {j}')
                window_size = int(len(line)*0.99)
                # print(window_size)
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    # print(output)
                    # print(len(torch.argsort(output, 1)[0]))
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        FP += 1
                        break
        time_check(f'=== Start evaluating anomaly data: {len(x_test_anomaly_gen)} ===')
        with torch.no_grad():
            for k, line in enumerate(x_test_anomaly_gen):        
                if k % 10000 == 0:
                    time_check(f'line {k}')
                window_size = int(len(line)*0.99)
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        TP += 1
                        break
        elapsed_time = time.time() - start_time
        time_check('elapsed_time: {:.3f}s'.format(elapsed_time))
        
        # Compute precision, recall and F1-measure
        FN = len(x_test_anomaly_gen) - TP
        alldata_len = len(x_test_normal_gen + x_test_anomaly_gen)
        ACC = 100 *((alldata_len-FN-FP)/(alldata_len))
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        RESULT = 'false positive (FP): {}, false negative (FN): {}, Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, ACC,P, R, F1)
        time_check(RESULT)
        time_check('Finished Predicting')
        filepath = f'{model_dir}/{log}_{ratio}_{target_filename}_evaulation_results.txt'
        with open(filepath, 'w') as file:
            file.write(f'{RESULT}\n')
        time_check(f'Results have been saved at: {filepath}')


if __name__ == '__main__':
    
    targetfile = 'bgl' # 'tbird' or 'spirit'
    # Create preprocessor for loading data
    preprocessor = Preprocessor(
        length  = 4,          
        timeout = float('inf'), # Do not include a maximum allowed time between events
    )
    # Load data from csv file
    if targetfile == 'bgl':
        X, y, labels, mapping = preprocessor.csv(f"../data_small/{targetfile}2.csv")
        
    elif targetfile == 'spirit' or targetfile == 'tbird':
        X, y, labels, mapping = preprocessor.csv(f"../data_small/{targetfile}_small_5m.csv")

    '''Hyperparameters'''
    num_epochs = 50
    num_classes = len(mapping)
    batch_size = 256
    input_size = 1

    # path to save model
    model_dir = './'
    num_layers = 2
    hidden_size = 64
    window_size = 3
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))  
    
    '''start training and testing the model'''
    run_model(target_filename=targetfile,window_size=window_size)

   