
from dataloader import read_target_data,get_data_csv,LogReader
import pandas as pd 


'''convert log in text format to csv format as required for Preprocessor.csv() in preprocess.py'''

if __name__ == '__main__':

    targetfile = 'bgl' # 'thunderbird' or 'spirit'
    log_payload,true_labels = read_target_data(targetfile)
    log_payload_deeplog = log_payload[['Timestamp','Content']]
    machine_name = log_payload.shape[0]*[targetfile]
    machine_df = pd.DataFrame({'machine':machine_name})
    # get two columns
    df = log_payload[['Timestamp','Content']]
    # rename above two columns
    df.rename(columns={'Timestamp': 'timestamp', 'Content': 'event'}, inplace=True)
    # add machinename column
    df.insert(loc=1,column ='machine',value=(log_payload.shape[0]*[targetfile]))
    # add labels column 
    df.insert(loc=3,column ='labels',value=true_labels)
    # save to csv file
    df.to_csv(f'../data_small/{targetfile}.csv',index=False)