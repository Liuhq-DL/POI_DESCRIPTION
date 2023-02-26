import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from collections import Counter
import json
from config.CONSTANTS import *


class Vocabulary():
    def __init__(self, vocab_path=None):
        self.id2word = {}
        self.word2id = {}
        self.index = 0
        self.counter = Counter()
        if vocab_path is not None and os.path.isfile(vocab_path):
            print("load vocabulary from: {}".format(vocab_path))
            self.load_from_path(vocab_path)
            
    def __call__(self, word):
        if word not in self.word2id:
            return self.word2id["<unk>"]
        return self.word2id[word]

    def __getitem__(self, item):
        return self.__call__(item)
    
    def __len__(self):
        return len(self.id2word)
    
    def id_to_word(self, word_index):
        if word_index not in self.id2word:
            raise ValueError
        return self.id2word[word_index]
    
    def check_load(self):
        assert len(self.word2id) == len(self.id2word)
        for i in self.word2id:
            assert i == self.id2word[self.word2id[i]]
        print("check load pass")
    
    def load_from_path(self, vocab_path):
        with open(vocab_path, "r", encoding=ENCODING) as f:
            vocab_data = json.load(f)
        self.word2id = vocab_data["word2id"]
        self.id2word = {index: word for word, index in self.word2id.items()}
        self.index = len(self.id2word)
        # self.check_load()
        
    def save_vocab(self, vocab_path):
        vocab_data = {}
        vocab_data["word2id"] = self.word2id
        if os.path.isfile(vocab_path):
            print("replace the existing vocab file: %s"%vocab_path)
        with open(vocab_path, "w", encoding=ENCODING) as f:
            json.dump(vocab_data, f)
    
    def collect_words(self, token_list):
        self.counter.update(token_list)
    
    def sing_insert(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.index
            self.id2word[self.index] = word
            self.index += 1

    def build(self,  min_num, max_len, rate=None, spec_word_list = ("<pad>", "<s>", "<\s>", "<unk>", "<sep>", "<rep>"),
                 vocab_path="./vocabulary1.json",
                 counter_path=None):
        # assert counter_path is not None
        if counter_path is not None:
            print("re-build from saved counter")
            with open(counter_path, "r", encoding=ENCODING) as f:
                self.counter = Counter(json.load(f))

        print("length of the counter: %d" % len(self.counter))
        if len(self.counter) < 10:
            raise RuntimeError
        for token in spec_word_list:
            self.sing_insert(token)
        sorted_words = self.counter.most_common()
        if rate is None:
            for i in range(min(max_len, len(self.counter))):
                if sorted_words[i][1] >= min_num:
                    self.sing_insert(sorted_words[i][0])
                else:
                    break
        else:
            print("build vocabulary based rate")
            acc_num = 0
            total_tokens = 0
            for token in sorted_words:
                total_tokens += token[1]
            if total_tokens == 0:
                raise ValueError

            for i in range(min(max_len, len(self.counter))):
                acc_num += sorted_words[i][1]
                self.sing_insert(sorted_words[i][0])
                if acc_num/total_tokens >= rate:
                    break

        self.save_vocab(vocab_path)
        print("build vocab length: ", len(self))

