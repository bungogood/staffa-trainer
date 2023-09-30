from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch import nn

from gnubg import GnuBGModel
from wildbg import WildBGModel

class Model(nn.Module, ABC):
    input_size: int
    dataset: Dataset

    @abstractmethod
    def to(self, device):
        raise NotImplementedError
