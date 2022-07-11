import numpy as np
import scipy as sp
from typing import Tuple, List

from src_lib import *


class Fusion_Model(Model):
    def __init__(self, n_classes: int, preProcess: PreProcess = PreProcess("None"),verbose: bool = False):
        super().__init__(n_classes, preProcess=preProcess)
        self.models : List[Model] = []