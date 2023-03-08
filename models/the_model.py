import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import numpy as np

import yaml
import tqdm
import random

from _BSS_hijack.models.the_net import SSSD


# scheduler (forward alpha and beta setting)
# p(x_t|x_s) => x_t = sqrt{alpha_t|s} * x_t + beta_t|s * epsilon
# p(x_t|x_0) => x_t = sqrt{_alpha_t} * x + _beta_t * epsilon
class SDEdit_sch():
    '''
    alpha and beta come from:   maybe
    Maximum Likelihood Training of Score-Based Diffusion Models
    '''
    def __init__(self, beta_min=0.1, beta_max=20) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max

    def ALPHA(self, t):
        # actually mean _alpha_t
        if isinstance(t, torch.Tensor):
            return torch.exp(-(self.beta_max - self.beta_min) * t**2 / 2 - t * self.beta_min)
        else:
            return np.exp(-(self.beta_max - self.beta_min) * t**2 / 2 - t * self.beta_min)

    def BETA(self, t):
        # _beta_t
        return 1 - self.ALPHA(t)
    
    def _beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def G(self, t):
        if isinstance(t, torch.Tensor):
            return torch.sqrt(self._beta(t))
        else:
            return np.sqrt(self._beta(t)) 
    
    def F(self, t):
        return - self._beta(t) / 2
    
    def disturb(self, x, t, z=None):
        # z = None ? 
        if isinstance(t, torch.Tensor):
            scale = self.ALPHA(t).sqrt()
            sigma = self.BETA(t).sqrt()
        else:
            scale = np.sqrt(self.ALPHA(t))
            sigma = np.sqrt(self.BETA(t))
        xt = scale * x
        z = torch.randn_like(x) if z == None else z
        xt += sigma * z
        return xt


# sampler
# the backward 
# p(x_t-1|x_t) => x_t-1 = MU(x_t) + SIGMA * epsilon
# in continuous form
# seems not work in the BSS problem
class Analytic_DPM():
    '''
    
    '''
    def __init__(self, scheduler):
        self.sch = scheduler
        # self.GAMMA = GAMMA
        # self.score = score_model

    def ALPHA_ts(self, t, s):
        return self.sch.ALPHA(t) / self.sch.ALPHA(s)

    def BETA_ts(self, t, s):
        return self.sch.BETA(t) - self.ALPHA_ts(t, s) * self.sch.BETA(s)

    def MU(self, score_net, x, t, s):
        dividend = self.ALPHA_ts(t, s)
        factor = 1 / torch.sqrt(dividend) if isinstance(dividend, torch.Tensor) else 1 / np.sqrt(dividend)
        grad = self.BETA_ts(t, s) * score_net(x, t)
        return factor * (x + grad)
    
    def SIGMA(self, GAMMA, t, s):
        factor = self.BETA_ts(t, s) / self.ALPHA_ts(t, s)
        return factor * (1 - self.BETA_ts(t, s) * GAMMA)
    

class BSS(nn.Module):
    # use score function
    # def __init__(self, device, data_channels=3) -> None:
    def __init__(self, args, config, device) -> None:
        super().__init__()
        self.args = args
        self.config = config
        
        self.device = device
        # setup 非参化
        self.scheduler = SDEdit_sch()   # setup 实例
        self.score_net = SSSD(nscheduler=self.scheduler)     # a score net
        self.sampler = Analytic_DPM(scheduler=self.scheduler,
        num_tokens=256, in_chn=3, mode='diag', measure='diag-lin', bidirectional=True).to(self.device)

        self.time_steps = 500
        # actually K steps, short sample

    def cal_loss(self, data):
        eps = 1e-5  # may no need
        B, C, L = data.shape
        # Batch size, Channel, Length
        
        random_t = torch.rand(B, device=self.device)
        # each batch use different t
        random_t = random_t[:, None, None]
        noise = torch.randn_like(data)
        # shape like data
        std = torch.sqrt(self.scheduler.BETA(random_t))
        # standard deviation

        perturbed_x = self.scheduler.disturb(data, random_t, noise)
        
        score = self.score_net(perturbed_x, random_t)
        
        loss = torch.mean(
            torch.sum((score * std + noise) + eps)**2, dim=(1, 2))

        return loss

    def sample(self, data_shape):
        eps = 1e-4  # replace 0

        init_point = 1  

        self.score_net.eval()

        init_x = torch.randn(
            data_shape, device=self.device) * np.sqrt(
                self.scheduler.BETA(init_point))
        # from random with variation 1
        trajectory = torch.linspace(init_point, eps, self.time_steps+1, device=self.device)
        K = trajectory.shape[0] # 
        x = init_x
        # traces = []
        # traces.append(x)

        with torch.no_gard():
            for t in tqdm.trange(K - 1):
                time_step = torch.ones(data_shape[0], device=self.device)[:, None, None]
                mean_x = self.sampler.MU(self.score_net, x, time_step * trajectory[t], time_step * trajectory[t+1])
                sigma = self.sampler.SIGMA(time_step * trajectory[t], time_step * trajectory[t+1])
                x = mean_x + sigma * torch.rand_like(x, device=self.device)
                # traces.append(mean_x)
        return mean_x # , traces

    # def FAST_sample(self, data):
    #     # **TEST** ref: Analytic_DPM
    #     return 1

    def BSS(self, perturbed_data, _lambda):
        eps = 1e-4
        init_point = 1
        
        # triple the perturbed data
        perturbed_data = perturbed_data * torch.ones([3, 1], device=self.device)
        data_shape = perturbed_data.shape

        self.score_net.eval()

        init_x = torch.randn(
            data_shape, device=self.device) * np.sqrt(
                self.scheduler.BETA(init_point))
        # from random with variation 1
        trajectory = torch.linspace(init_point, eps, self.time_steps+1, device=self.device)
        K = trajectory.shape[0] # 
        x = init_x
        # traces = []
        # traces.append(x)

        with torch.no_gard():
            for t in tqdm.trange(K - 1):
                time_step = torch.ones(data_shape[0], device=self.device)[:, None, None]
                mean_x = self.sampler.MU(self.score_net, x, time_step * trajectory[t], time_step * trajectory[t+1])
                sigma = self.sampler.SIGMA(time_step * trajectory[t], time_step * trajectory[t+1])
                
                y_t = mean_x * perturbed_data 
                x_prime = mean_x + (_lambda / 3) * (torch.ones([3, 1]) * y_t - torch.eye(3) * y_t)
                
                x = mean_x + sigma * torch.rand_like(x, device=self.device)
                # traces.append(mean_x)
        return mean_x # , traces

    def forward(self, data):
        return self.cal_loss(data)
