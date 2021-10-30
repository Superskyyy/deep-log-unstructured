"""
Do not run this script in Windows, it will not work due to use of signal
"""
import os
import pickle
import re

import signal


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

import numpy as np
from tqdm.auto import tqdm

from data_tokenizer import DataTokenizer

tokenizer = DataTokenizer()


class DataImporter:
    """
    loads data set from the raw dataset
    """

    def __init__(self, log_template, dataset_folder_path, dataset_name, dataset_step=1,
                 dataset_limit=100000, dataset_type='main', normal_indicator: str = '-', aux_count=50000,
                 chunk: bool = True):
        self.log_template = log_template  # a template containing <Token{n}> and <Message>
        self.log_dataframe = None
        self.dataset_folder_path: str = dataset_folder_path  # path to the dataset folder
        self.dataset_name: str = dataset_name  # full name of raw dataset
        self.step: int = dataset_step  # step taken to sample auxiliary dataset
        self.log_template_regex: re = re.compile(r'')
        self.log_template_headers: list[str] = []
        self.limit: int = dataset_limit  # used for faster experiment only
        self.dataset_type: str = dataset_type
        self.normal_indicator: str = normal_indicator  # a sign indicating the log line is anomaly
        self.aux_count: int = aux_count
        self.chunk: bool = chunk

    def log_loader(self):
        """
        read from IO stream and only take the actual log message based on template
        :return:
        """
        log_messages = []
        true_labels = []
        with open(os.path.join(self.dataset_folder_path, self.dataset_name), 'r') as ds:
            for line_no, line in enumerate(tqdm(ds, miniters=1)):
                if line_no != 1 and line_no % 20000000 == 1:  # 15 file chunks
                    with open(f'dataset/tbird/{self.dataset_name}_chunked_msg_line{line_no}', 'wb') as message_file:
                        # print(log_messages)
                        # log_messages_array = np.asanyarray(log_messages).reshape(-1, 1)
                        print(len(log_messages))
                        l = len(log_messages)
                        tokenized = [tokenizer.tokenize(log_message) for log_message in
                                     tqdm(log_messages, position=0, leave=True, total=l)]
                        tokenized_np = np.asanyarray(tokenized)
                        del tokenized
                        pickle.dump(tokenized_np, message_file)
                        log_messages = []  # reset
                # Start the timer. Once 10 seconds are over, a SIGALRM signal is sent.
                signal.alarm(10)
                try:

                    match = self.log_template_regex.search(line.strip())

                    if not match:
                        continue
                    label_decider = lambda x: 0 if x == self.normal_indicator else 1
                    true_labels.append(label_decider(match.group('Token0')))
                    # message = tokenizer.tokenize(match.group('Message'))
                    # log_messages.append(message)
                    log_messages.append(match.group('Message'))
                    # print('message after tokenize ', message, log_messages)
                    # print(self.log_template_headers)
                except TimeoutException:
                    print("Regex hang detected, skipping")
                    continue  # catastrophic backtracking
                except Exception as e:  # noqa
                    print(e)
                    # print(e) # will skip those without normal indications('-' OR 'warn')
                    pass
                else:
                    signal.alarm(0)
                if line_no == self.limit:
                    break
        return log_messages, np.array(true_labels)  # remaining log_messages and all of labels

    def load(self):
        self.log_template_matcher()

        self.log_dataframe = self.log_loader()

        # if self.dataset_type == 'auxiliary':
        #     print(log_messages.iloc[true_labels.flatten() == 0].shape)
        #     print(log_messages.iloc[true_labels.flatten() == 0])
        #     df_normal = log_messages.iloc[true_labels.flatten() == 0].sample(n=self.aux_count).values
        #     df_anomalies = log_messages.iloc[true_labels.flatten() == 1].sample(n=self.aux_count).values
        #     return df_normal, df_anomalies
        # elif self.dataset_type == 'main':
        #     return log_messages, true_labels

    def log_template_matcher(self):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', self.log_template)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.+?)' % header
                headers.append(header)
        print(regex)
        regex = re.compile('^' + regex + '$')

        self.log_template_headers, self.log_template_regex = headers, regex
