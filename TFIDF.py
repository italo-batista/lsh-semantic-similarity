# coding: utf-8

import math
import pandas as pd
import numpy as np
import nltk
import re
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from time import time
from textblob import TextBlob as tb


class TFIDF(object):

    def __init__(self, texts, langue):

        nltk.download("rslp")

        self.texts = texts
        self.langue = langue
        self._list_extracted_values = []
        self._row_axis, self._col_axis = 0, 1

        self._df = pd.DataFrame(
            data={'TEXT': self.texts, 'ID': range(len(self.texts))})

    def _to_lower_case(self, text):

        text = text.lower()
        text = text.replace('á', 'a')
        text = text.replace('é', 'e')
        text = text.replace('í', 'i')
        text = text.replace('ó', 'o')
        text = text.replace('ú', 'u')
        text = text.replace('à', 'a')
        text = text.replace('ã', 'a')
        text = text.replace('õ', 'o')
        text = text.replace('ç', 'c')
        text = text.replace('â', 'a')
        text = text.replace('ê', 'e')
        text = text.replace('ô', 'o')

        return text

    def _remove_stopwords(self, text):

        pt_stop = get_stop_words(self.langue)
        self._tokenizer = RegexpTokenizer(r'\w+')

        tokens = self._tokenizer.tokenize(text)
        stopped_tokens = list(filter(lambda t: t not in pt_stop, tokens))
        without_stopwords = " ".join(stopped_tokens)

        return without_stopwords

    def _tf(self, word, blob):
        return blob.words.count(word) / len(blob.words)

    def _n_containing(self, word, bloblist):
        return len(list(filter(lambda blob: word in blob, bloblist)))

    def _idf(self, word, bloblist):
        n_textos = len(bloblist)
        return math.log(n_textos / (1 + self._n_containing(word, bloblist)))

    def _tfidf(self, word, blob, bloblist):
        return self._tf(word, blob) * self._idf(word, bloblist)

    def _get_text_radicals(self, text):

        stemmer = nltk.RSLPStemmer()
        blob = tb(text)
        radicals = [stemmer.stem(word) for word in blob.words]
        text_radicals = " ".join(radicals)

        return text_radicals

    def _get_important_radicals(self, text):
        blob = tb(text)
        scores = {radical: self._tfidf(radical, blob, self.columns_radicals)
                  for radical in blob.words}
        median_score = np.median(list(scores.values()))
        important_radicals = list(
            filter(lambda x: x[1] >= median_score, scores.items()))
        return important_radicals

    def _extract_tokens(self):

        self._df['TEXT'] = self._df.apply(
            lambda row: self._to_lower_case(row['TEXT']), axis=self._col_axis)

        without_stopwords = self._df['TEXT'].apply(
            lambda x: self._remove_stopwords(x)
        )
        self._df['WITHOUT_STOPWORDS'] = without_stopwords

        self._df['TEXT_OF_RADICALS'] = self._df['WITHOUT_STOPWORDS'].apply(
            lambda x: self._get_text_radicals(x)
        )
        self.columns_radicals = list(self._df['TEXT_OF_RADICALS'])

        important_radicals = self._df['TEXT_OF_RADICALS'].apply(
            lambda text: self._get_important_radicals(text)
        )
        self._df['IMPORTANT_RADICALS'] = important_radicals.apply(
            lambda radicals_scores: [r for r, s in radicals_scores]
        )

        aux = self._df
        self._df = pd.DataFrame(data={})
        self._df['IMPORTANT_RADICALS'] = aux['IMPORTANT_RADICALS']

    def _conc_my_tokens(self, tokens, i):
        if type(tokens) == list:
            self.tokens_texts[i] += tokens

    def get_sparse_matrix(self):

        self._extract_tokens()

        words_set = []
        for l in list(self._df['IMPORTANT_RADICALS']):
            words_set += l

        words_set = list(set(words_set))

        self.tokens_texts = {i: [] for i in range(len(self.texts))}
        for i in range(len(self.texts)):
            self._df.apply(lambda row: self._conc_my_tokens(
                row[i], i), axis=self._row_axis)

        matrix = [[None] * len(words_set) for i in range(len(self.texts))]
        for text in range(len(self.texts)):
            for i, token in enumerate(words_set):
                has = 1 if token in self.tokens_texts[text] else 0
                matrix[text][i] = has

        return matrix
