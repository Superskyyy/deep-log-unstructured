import preprocessing, utils, models
from utils import time_check
import numpy as np
import pandas as pd
import models
from  dataloader import read_auxiliary_data, read_target_data, train_test_split

# torch.random.seed = 0
np.random.seed(0)

def run_model(embedding_type, x_train, y_train, x_test, y_test, log_type,train_ratio=0.8):
    '''train and evaluate the model
    
    return: 
         a list format of -  target dataset name, the preprocessing technique, model name, size of training data ratio,
                             , anamoly detection threshold, accuracy, precison, recall, and F1 score
    '''
    results = []
    print('Evaluating {} on log type {} with train/test ratio {}/{}:'.format('PCA', log_type,
                                                                  train_ratio, (1 - train_ratio)))
    model = models.PCA()
    model.fit(x_train[y_train == 0]) 
    acc,precision, recall, f1 = model.evaluate(x_test, y_test)
    results = [log_type, embedding_type, 'PCA', train_ratio, model.threshold, acc, precision, recall, f1]
    return results

if __name__ == '__main__':
    
    '''Read and preprocess data using TF-IDF
    
       Train the model on preprocessed data
       
       the results are saved in a csv format
    
    '''
    
    # specify to evaluate which datasets:
    targetfile = 'thunderbird'# 'spirit', 'bgl'
    
    # read auxiliary data 
    aux_anomalies0 = read_auxiliary_data()
    # read target data and append it with auxiliary data 
    concat_data,concat_labels,df_size = read_target_data(aux_anomalies0,targetfile)
    
    # a variable control to save the data
    savepath = True

    saved_results_path = f'../outputs/PCA_{targetfile}_datalen_{len(concat_data)}_result.csv'
    time_check(f'Start training and evaluating')
    # start training
    train_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    for ratio in (train_ratios):
        x_train,y_train,x_test,y_test = train_test_split(concat_data,concat_labels,ratio,df_size)
        time_check(f'Evaluating on train_ratio = {ratio}')
        feature_extractor = preprocessing.FeatureExtractorTFIDF()
        x_train_flatten = feature_extractor.fit_transform(x_train.flatten())
        x_test_flatten = feature_extractor.transform(x_test.flatten())
        current_results= run_model('tf-idf', x_train_flatten, y_train, x_test_flatten, y_test,
                                       log_type=targetfile,train_ratio=ratio)
        
        # save the results 
        if savepath:
            pd.DataFrame([current_results], columns=['Log_Type', 'Embedding_Type', 'Model', 'Train_Ratio', 'Threshold',
                                     'Accuracy', 'Precision', 'Recall', 'F1']).to_csv(saved_results_path, mode='a',index=False,header=False)
        time_check(f'Finished evaluating on train_ratio = {ratio}')

    






