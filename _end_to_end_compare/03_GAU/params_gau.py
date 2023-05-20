
import numpy as np
import torch

class AttrDict(dict):
    # 用 ‘.’ 来调用 dict 里的值
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
    
    
params = AttrDict(
    batch_size=64,
    learning_rate=1e-4,

    residual_layers=5,

    dim = 8,
    hidden_dim = 16,
    query_key_dim=64,

    device = torch.device('cuda:2'),
)