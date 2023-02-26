import jieba
import os
import sys
from config.CONSTANTS import *


class Tokenlize():
    def __init__(self, tokenlizer = "jieba"):
        print("use original tokenizer")
        jieba.initialize()
        if tokenlizer == "jieba":
            self.cut = jieba.cut
        else:
            raise RuntimeError

    def __call__(self, des_text):
        if PRE_TOKENIZE:
            return des_text.split("#")
        return list(self.cut(des_text))


class SynTokenlize():
    def __init__(self, tokenlizer = "jieba"):
        print("use synthesize tokenizer")
        jieba.initialize()
        if tokenlizer == "jieba":
            self.cut = jieba.cut
        else:
            raise RuntimeError

    def __call__(self, des_text, pre_token):
        if not pre_token:
            return list(self.cut(des_text))
        else:
            return des_text.split("#")
