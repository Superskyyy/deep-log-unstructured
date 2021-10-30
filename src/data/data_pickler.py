import os
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange

from data_importer import DataImporter
from data_tokenizer import DataTokenizer

folder_path = 'dataset'


def pickle_import(template, name, normal_indicator,folder):
    """
    This is only for main data not aux

    """
    message_remained, label = DataImporter(log_template=template, dataset_folder_path=folder_path,
                                  dataset_name=name, dataset_step=1, dataset_type='main',
                                  dataset_limit=999999999999, normal_indicator=normal_indicator, aux_count=5000).load()
    print(
        f'\nSuccessfully imported - remaining {len(message_remained)} => of all {len(label)} Messages from dataset at {os.path.join(folder_path, name)}\n\n')

    with open(f'dataset/{folder}/{name}_chunked_msg_line_final_batch', 'wb') as message_file:
        pickle.dump(message_remained, message_file)
    with open(f'dataset/{folder}/{name}_full_label{len(label)}', 'wb') as label_file:
        pickle.dump(label, label_file)
    # return message, label


def pickle_tokenize(template, name, normal_indicator):
    tokenizer = DataTokenizer()
    pickle_import(template=template, name=name, normal_indicator=normal_indicator)
    # above pickles the imported message and labels, message is further passed down to tokenize

    log_messages = log_messages.values.reshape(-1, 1)
    print("\n\n##################### Data Shape ##################")
    print(f"log message shape {log_messages.shape}, log label shape {labels.shape}")
    print("##################### Data Shape End ##############")
    df_len = int(log_messages.shape[0])
    data_tokenized = []  # don't append to numpy array, inefficient
    for i in trange(df_len, miniters=1):
        tokenized = tokenizer.tokenize(log_messages[i][0])
        data_tokenized.append(tokenized)  # type: list

    print(data_tokenized)
    data_tokenized = np.asanyarray(data_tokenized)
    print(data_tokenized)
    data_tokenized_padded = pad_sequences(data_tokenized, maxlen=50, truncating="post", padding="post")

    with open(f'{name}_full_message_tokenized', 'wb') as tokenized_file:
        pickle.dump(data_tokenized_padded, tokenized_file)


# bgl_template = '<Token0> <Token1> <Token2> <Token3> <Token4> <Token5> <Token6> <Token7> <Token8> <Message>'
#
# pickle_tokenize(template=bgl_template, name='bgl2', normal_indicator='-')

thunderbird_template = '<Token0> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Message>'  # thunderbird

pickle_import(template=thunderbird_template, name='tbird2', normal_indicator='-',folder='tbird')
