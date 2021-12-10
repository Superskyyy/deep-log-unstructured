import numpy as np
import pandas as pd
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
        
        
class LogTokenizer:
    def __init__(self):
        self.word2index = {'[PAD]':0, '[CLS]':1, '[MASK]':2}
        self.index2word = {0:'[PAD]', 1:'[CLS]', 2:'[MASK]'}
        self.n_words = 3  # Count SOS and EOS
        self.stop_words = set(stopwords.words('english'))
        self.regextokenizer =  nltk.RegexpTokenizer('\w+|.|')
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def tokenize(self, sent):
        sent = re.sub(r'\/.*:', '', sent, flags=re.MULTILINE)
        sent = re.sub(r'\/.*', '', sent, flags=re.MULTILINE)
        sent = self.regextokenizer.tokenize(sent)
        sent = [w.lower() for w in sent]
        sent = [word for word in sent if word.isalpha()]
        sent = [w for w in sent if not w in self.stop_words]
        sent = ['[CLS]'] + sent
        for w in range(len(sent)):
            self.addWord(sent[w])
            sent[w] = self.word2index[sent[w]]
        return sent
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word2index[w] for w in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.index2word[i] for i in ids]



    
def get_data_csv(log_file, input_dir, output_dir, log_format, regex=[], every_n=10, aux=0, max_lines=5000000):
    reader = LogReader(log_format, log_file, indir=input_dir, outdir=output_dir, rex=regex, every_n=every_n, max_lines=max_lines)
    reader.load_data()
    # log_payload, true_labels = reader.df_log.Content, np.where(reader.df_log.t.values=='-',0,1)
    log_payload, true_labels = reader.df_log, np.where(reader.df_log.t.values=='-',0,1)

    return log_payload, true_labels




def read_target_data(targetfile):
    
    '''Read both normal and anomaly data from target dataset based on the 
        targetfile = 'bgl' or 'spirit' or 'thunderbird'
               
       
       Return: 
       
        log_payload = concatenated data from target datasets 
        
        true_labels = concatenated data labels from target datasets 
        
    
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
        log_payload, true_labels = get_data_csv(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)

        
    elif targetfile == 'spirit':
        #read the target dataset
        log_file = 'spirit2_small_5m'
        log_format = log_format_spirit
        # aux = 0 means read current dataset as target dataset
        aux_size = 0
        log_payload, true_labels = get_data_csv(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)
    elif targetfile == 'thunderbird':
        #read the target dataset
        log_file = 'tbird2_small_5m'
        log_format = log_format_tbird 
        # aux = 0 means read current dataset as target dataset
        aux_size = 0
        log_payload, true_labels = get_data_csv(log_file, input_dir, output_dir, log_format, regex, every_n=every_n,
                                            aux=aux_size, max_lines=5000000)
    
    return log_payload,true_labels



def train_test_split(x,y,labels,ratio):
    
    '''split data for train set and test set'''
    train_size = int(len(x) * ratio)
    test_size = int(len(y) * (1-ratio))
    print('Train size: ',train_size)
    print('Test size: ',test_size)
    
    '''train set'''
    x_train = x[:train_size]#[train_normal_index]
    y_train = y[:train_size]#[train_normal_index]
  
    print(f'x_train.shape = {x_train.shape}')
    # print(f'x_train.shape = {x_train.shape}')  
    '''test set'''
    x_test = x[train_size:train_size+test_size]
    y_test = y[train_size:train_size+test_size]
    
    print(f'x_test.shape = {x_test.shape}')
    '''labels'''
    
    label_train = labels[:train_size]
    label_test =  labels[train_size:train_size+test_size]
    
    # print(f'x_train.shape = {x_train.shape}')    
    return x_train,y_train,x_test,y_test,label_train,label_test
 