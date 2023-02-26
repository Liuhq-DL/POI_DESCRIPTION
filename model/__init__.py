import os
import sys
dir_name = os.path.abspath(os.path.join(os.path.dirname("__name__"), "../"))
sys.path.append(dir_name)
from config.CONSTANTS import *

from torch.nn import CrossEntropyLoss
from data.pre_func import SynTokenlize

from model.guider_selector_model import GSModel, GSProcess

