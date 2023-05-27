import numpy as np


class AttrDict(dict):
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

params_diffwave_cat = AttrDict(
    embed_dim = 256,
    residual_layers=24,
    residual_channels=12,
    dilation_cycle_length=8,
)

params_diffwave3 = AttrDict(
    # Model params
    embed_dim = 256,
    residual_layers=32,
    residual_channels=36,
    dilation_cycle_length=10,
)

params_gau_cat = AttrDict(
    embed_dim = 256,
    
    residual_layers=24,
    dim=8,
    hidden_dim=16,
    query_key_dim=128,
)

params_gau3 = AttrDict(
    embed_dim = 256,
    
    residual_layers=24,
    dim=24,
    hidden_dim=36,
    query_key_dim=128,
)