import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractorTFIDF(object):
    '''TF-IDF preprocessing'''

    def __init__(self, stemming=False, max_features=500):
        self.stop_words = set(stopwords.words('english'))
        self.regextokenizer = nltk.RegexpTokenizer('\w+|.|')
        self.stemming = None
        if stemming:
            self.stemming = PorterStemmer()
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def __tokenize(self, x_i):
        sent = re.sub(r'\/.*:', '', x_i, flags=re.MULTILINE)
        sent = self.regextokenizer.tokenize(sent)
        filtered_sent = []
        for i, w in enumerate(sent):
            if w.isalpha() and w not in self.stop_words:
                if self.stemming:
                    filtered_sent.append(self.stemming.stem(w.lower()))
                else:
                    filtered_sent.append(w.lower())
        return filtered_sent

    def __preprocess(self, x):
        lines_processed = []
        for x_i in x:
            lines_processed.append(" ".join(self.__tokenize(x_i)))
        return lines_processed

    def fit_transform(self, x):
        print('====== Transformed train data summary ======')
        x = self.__preprocess(x)
        vectors = self.vectorizer.fit_transform(x).toarray()

        print('Train data shape: ({}, {})\n'.format(vectors.shape[0], vectors.shape[1]))
        return vectors

    def transform(self, x):
        print('====== Transformed test data summary ======')
        x = self.__preprocess(x)
        vectors = self.vectorizer.transform(x).toarray()

        print('Test data shape: ({}, {})\n'.format(vectors.shape[0], vectors.shape[1]))
        return vectors
