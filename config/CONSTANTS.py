import os

USE_NOHUP = False

PRE_TOKENIZE = True
print("pre tokenlize: ", PRE_TOKENIZE)

DATA_ROOT_PATH = ""

PRETRAINED_ROOT = ""

ENCODING = "utf-8"

EPSILON_FILTER = 0.0001

START_TOKEN = "<s>"
END_TOKEN = "<\s>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
REPRESENT_TOKEN = "<rep>"

SPECIAL_TOKEN_LIST = ["<pad>", "<s>", "<\\s>", "<unk>", "<sep>", "<rep>"]

START_INDEX = SPECIAL_TOKEN_LIST.index(START_TOKEN)
END_INDEX = SPECIAL_TOKEN_LIST.index(END_TOKEN)
UNK_INDEX = SPECIAL_TOKEN_LIST.index(UNK_TOKEN)
PAD_INDEX = SPECIAL_TOKEN_LIST.index(PAD_TOKEN)
SEP_INDEX = SPECIAL_TOKEN_LIST.index(SEP_TOKEN)
REPRESENT_INDEX = SPECIAL_TOKEN_LIST.index(REPRESENT_TOKEN)

PRE_END_TOKEN_LIST = ["。", "！", "；"]
ENG_PRE_END_TOKEN_LIST = [".", "!", ";"]

MODEL_NAME = "model-iter-{:0>8d}.pt"
OPTIMIZER_NAME = "optimizer-iter-{:0>8d}.pt"
BEST_MODEL_MANE = "best_model.pt"

SEARCH_RADIOUS = 2000


