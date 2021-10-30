import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

"""
Standard tokenizer + do what the paper says
"""


class DataTokenizer:
    def __init__(self):
        self.word2index = {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2}
        self.num_words = 3
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, message):
        # paper section IV: Tokenization processing
        message = message.lower()
        message = re.sub(r'/.*:', '', message, flags=re.MULTILINE)  # filter for endpoints
        message = re.sub(r'/.*', '', message, flags=re.MULTILINE)
        message = word_tokenize(message)  # remove non words
        message = (word for word in message if word.isalpha())  # generator  # remove numerical
        message = [word for word in message if word not in self.stop_words]  # remove nltk common stopwords
        message = ['[CLS]'] + message  # add embedding token
        for word_idx, word in enumerate(message):  # convert to value
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.num_words += 1
            message[word_idx] = self.word2index[word]
        return message
