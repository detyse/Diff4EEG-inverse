import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import numpy as np

import yaml
import tqdm
import random
from utils.utils import type_align

from models.SSSD import SSSD


class BSSmodel(nn.Module):
    # use score function
    # def __init__(self, device, data_channels=3) -> None:
    def __init__(self, args, config) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        # setup 非参化
        self.scheduler = Score_sch()   # setup 实例
        self.score_net = SSSD(nscheduler=self.scheduler, 
                              config=self.config, 
                              mode=config.s4.mode, 
                              measure=config.s4.measure).to(self.device)
        self.sampler = VDM_sampler(scheduler=self.scheduler)
        self.time_steps = args.timesteps    # 把timesteps设为args方便修改
        # actually K steps, short sample

    def cal_loss(self, data):
        eps = 1e-5  # may no need
        # input perturbed data have 1 channel
        data = type_align(data).to(self.device)
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
            torch.sum((score * std + noise) + eps, dim=(1, 2))**2)

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
    

    def hijack_sample(self, perturbed_data, _lambda):
        eps = 1e-4
        init_point = 1
        
        # triple the perturbed data
        perturbed_data = type_align(perturbed_data).to(self.device)
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
    


# scheduler (forward alpha and beta setting)
# 1 >= t > s >= 0
# p(x_t|x_s) => x_t = sqrt{alpha_t|s} * x_t + sqrt{beta_t|s} * epsilon
# p(x_t|x_0) => x_t = sqrt{_alpha_t} * x + sqrt{_beta_t} * epsilon
class Score_sch():
    '''
    alpha and beta come from:   maybe
    ref: Maximum Likelihood Training of Score-Based Diffusion Models
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
        # z could be Other noise that is not Gaussian
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
class VDM_sampler():
    def __init__(self, scheduler) -> None:
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
    
    def SIGMA(self, t, s):
        # t > s
        sigma = self.sch.BETA(s) / self.sch.BETA(t) * self.BETA_ts(t, s)
        return torch.sqrt(sigma) if isinstance(sigma, torch.Tensor) else np.sqrt(sigma)


# class VariationalDMSampler():
#     """
#     ref:
#     Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). 
#     SDEDIT: GUIDED IMAGE SYNTHESIS AND EDITING WITH STOCHASTIC DIFFERENTIAL EQUATIONS. 33.
#     P21 Algorithm 4
#     Li, C., Zhu, J., & Zhang, B. (2022). 
#     ANALYTIC-DPM: AN ANALYTIC ESTIMATE OF THE OPTIMAL REVERSE VARIANCE IN DIFFUSION PROB- ABILISTIC MODELS. 39.
#     P22
#     Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models.
#     arXiv preprint arXiv:2107.00630, 2021.
#     """

#     def __init__(self, scheduler):
#         self.sch = scheduler

#     def ALPHA_ts(self, t, s):
#         return self.sch.ALPHA(t) / self.sch.ALPHA(s)
    
#     def BETA_ts(self, t, s):
#         return self.sch.BETA(t) - self.ALPHA_ts(t, s) * self.sch.BETA(s)

#     def MU(self, score_model, x, cond_info, t, s):
#         # t > s
#         dividend = self.ALPHA_ts(t, s)
#         factor = 1 / torch.sqrt(dividend) if isinstance(dividend, torch.Tensor) else 1 / np.sqrt(dividend)
#         grad = self.BETA_ts(t, s) * score_model(x, cond_info, t)
#         return factor * (x + grad)
    
#     def SIGMA(self, t, s):
#         # t > s
#         sigma2 = self.sch.BETA(s) / self.sch.BETA(t) * self.BETA_ts(t, s)
#         return torch.sqrt(sigma2) if isinstance(sigma2, torch.Tensor) else np.sqrt(sigma2)

# sampler
# the backward 
# p(x_t-1|x_t) => x_t-1 = MU(x_t) + SIGMA * epsilon
# in continuous form
# seems not work in the BSS problem
# class Analytic_DPM():
#     '''
#     could not use in this project, which require a known diffusion path.
#     '''
#     def __init__(self, scheduler):
#         self.sch = scheduler
#         # self.GAMMA = GAMMA
#         # self.score = score_model

#     def ALPHA_ts(self, t, s):
#         return self.sch.ALPHA(t) / self.sch.ALPHA(s)

#     def BETA_ts(self, t, s):
#         return self.sch.BETA(t) - self.ALPHA_ts(t, s) * self.sch.BETA(s)

#     def MU(self, score_net, x, t, s):
#         dividend = self.ALPHA_ts(t, s)
#         factor = 1 / torch.sqrt(dividend) if isinstance(dividend, torch.Tensor) else 1 / np.sqrt(dividend)
#         grad = self.BETA_ts(t, s) * score_net(x, t)
#         return factor * (x + grad)
    
#     def SIGMA(self, GAMMA, t, s):
#         factor = self.BETA_ts(t, s) / self.ALPHA_ts(t, s)
#         return factor * (1 - self.BETA_ts(t, s) * GAMMA)
