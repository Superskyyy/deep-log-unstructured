import os
import re

import numpy as np
import pandas
from tqdm import tqdm


class DataImporter:
    """
    loads data set from the raw dataset
    """

    def __init__(self, log_template, dataset_folder_path, dataset_name, dataset_step=1,
                 dataset_limit=100000, dataset_type='main', normal_indicator: str = '-', aux_count=50000):
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

    def log_loader(self):
        """
        read from IO stream and only take the actual log message based on template
        :return:
        """
        log_messages = []
        counter = 0

        with open(os.path.join(self.dataset_folder_path, self.dataset_name), 'r') as ds:
            for line_no, line in enumerate(tqdm(ds, miniters=1)):
                if line_no % self.step == 0:
                    try:
                        match = self.log_template_regex.search(line.strip())
                        message = [match.group(header) for header in self.log_template_headers]
                        log_messages.append(message)
                        counter += 1
                    except Exception as e:  # noqa
                        pass
                if line_no == self.limit:
                    break
        df = pandas.DataFrame(log_messages, columns=self.log_template_headers)
        df.insert(0, 'LineId', None)
        df['LineId'] = [i + 1 for i in range(counter)]
        return df

    def load(self):
        self.log_template_matcher()

        self.log_dataframe = self.log_loader()

        # differentiate anomaly with normal log
        log_messages = self.log_dataframe.Message
        true_labels = np.where(self.log_dataframe.Token0.values == self.normal_indicator, 0, 1)

        if self.dataset_type == 'auxiliary':
            print(log_messages.iloc[true_labels.flatten() == 0].shape)
            print(log_messages.iloc[true_labels.flatten() == 0])
            df_normal = log_messages.iloc[true_labels.flatten() == 0].sample(n=self.aux_count).values
            df_anomalies = log_messages.iloc[true_labels.flatten() == 1].sample(n=self.aux_count).values
            return df_normal, df_anomalies
        elif self.dataset_type == 'main':
            return log_messages, true_labels

    def log_template_matcher(self):
        headers = []
        template_chunks = re.split(r'(<[^<>]+>)', self.log_template)
        expression = ''
        for template_chunk_idx in range(len(template_chunks)):
            if template_chunk_idx % 2 == 0:
                splitter = re.sub(' +', '\\\s+', template_chunks[template_chunk_idx])
                expression += splitter
            else:
                header = template_chunks[template_chunk_idx].strip('<').strip('>')
                expression += '(?P<%s>.*?)' % header
                headers.append(header)
        print(expression)
        expression = re.compile('^' + expression + '$')

        self.log_template_headers, self.log_template_regex = headers, expression
