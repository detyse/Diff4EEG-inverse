# SSSD params

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
    in_chn = 1,
    num_tokens = 32,
    depth = 24,

    mode = 'diag',
    measure = 'diag-lin',
    bidirectional = True,
    transpose = False,

    device = torch.device('cuda:2'),
)