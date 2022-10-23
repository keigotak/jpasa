# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict, OrderedDict


class Indexer:
    def __init__(self):
        self.index = dict()
        self.inverse_index = dict()
        self.freq = OrderedDict()

    def fit(self, indexes, threshold=-1):
        indexes = np.ravel(indexes)
        freq = defaultdict(int)

        for word in indexes:
            freq[word] += 1
        self.freq = OrderedDict(sorted(freq.items(), reverse=True, key=lambda x: x[1]))

        number = 0
        for word in self.freq.keys():
            if threshold == -1:
                self.index[word] = number
                number += 1
            else:
                if self.freq[word] > threshold:
                    self.index[word] = number
                    number += 1
                else:
                    break

        for word, id in self.index.items():
            self.inverse_index[id] = word

    def transform_sentences(self, ids):
        ids_ret = []
        for id in ids:
            ids_ret.append([self.index[item] if item in self.index.keys() else self.get_unk_id() for item in id])
        return np.array(ids_ret)

    def get_unk_word(self):
        return "<unk>"

    def get_pad_word(self):
        return "<pad>"

    def get_null_word(self):
        return "<null>"

    def get_unk_id(self):
        return len(self.index)

    def get_pad_id(self):
        return len(self.index) + 1

    def get_null_id(self):
        return len(self.index) + 2

    def __len__(self):
        return len(self.index) + 3

    def id2word(self, index):
        ret = self.get_unk_word()
        index = int(index)
        if index in self.inverse_index.keys():
            ret = self.inverse_index[index]
        else:
            if index == self.get_unk_id():
                ret = self.get_unk_word()
            elif index == self.get_pad_id():
                ret = self.get_pad_word()
            elif index == self.get_null_id():
                ret = self.get_null_word()
        return ret

    def word2id(self, word):
        if word == self.get_unk_word():
            return self.get_unk_id()
        elif word == self.get_pad_word():
            return self.get_pad_id()
        elif word == self.get_null_word():
            return self.get_null_id()

        if word in self.index.keys():
            return self.index[word]
        return None


